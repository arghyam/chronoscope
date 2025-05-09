import glob
import os

import cv2
import numpy as np
from ultralytics import YOLO

# Load models once at the beginning
broader_meter_model = YOLO("runs_broader_meter/obb/train/weights/best.pt")
individual_numbers_model = YOLO("runs_indi_full_10k/obb/train/weights/best.pt")


def straighten_box(image, bbox_points):
    """
    Straighten an oriented bounding box region using perspective transformation
    without flipping the image orientation

    Args:
        image: Original image (numpy array)
        bbox_points: List/array of 4 points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                    representing the oriented bounding box corners

    Returns:
        Straightened image of the box region
    """
    # Convert points to numpy array
    pts = np.array(bbox_points, dtype=np.float32)

    # We need to determine the correct ordering of points for our rectangle
    # Compute the center of the bounding box
    center = np.mean(pts, axis=0)

    # For each point, compute the angle to the center
    angles = []
    for pt in pts:
        angle = np.arctan2(pt[1] - center[1], pt[0] - center[0])
        angles.append(angle)

    # Sort points based on their angle to get them in clockwise/counter-clockwise order
    sorted_indices = np.argsort(angles)
    pts = pts[sorted_indices]

    # Calculate width and height
    width = int(max(
        np.linalg.norm(pts[0] - pts[1]),
        np.linalg.norm(pts[2] - pts[3])
    ))

    height = int(max(
        np.linalg.norm(pts[1] - pts[2]),
        np.linalg.norm(pts[3] - pts[0])
    ))

    # Define destination points for the perspective transform
    dst_pts = np.array([
        [0, 0],               # top-left
        [width - 1, 0],       # top-right
        [width - 1, height - 1], # bottom-right
        [0, height - 1]       # bottom-left
    ], dtype=np.float32)

    # Calculate perspective transform matrix
    M = cv2.getPerspectiveTransform(pts, dst_pts)

    # Apply perspective transformation
    warped = cv2.warpPerspective(image, M, (width, height))

    return warped


def enhance_image(image):
    """
    Enhance the image to improve readability for OCR while preserving color
    """
    # Create a copy of the original image
    enhanced = image.copy()

    # Apply bilateral filtering to reduce noise while preserving edges
    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)

    # Increase contrast and brightness
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=15)

    # Apply subtle sharpening
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)

    # Optional: Reduce glare (if present in some images)
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced


def sort_boxes_by_position(boxes, classes):
    """
    Sort detected digit boxes by their x-coordinate for left-to-right reading order.
    Works with oriented bounding boxes in polygon format.

    Args:
        boxes: List of bounding box coordinates in polygon format
        classes: List of corresponding class labels

    Returns:
        Tuple of (sorted boxes, sorted classes)
    """
    # Create a list of (box, class) tuples
    box_class_pairs = list(zip(boxes, classes))

    # For oriented bounding boxes, use the leftmost x-coordinate of each box
    # This works better for digit readings that might be slightly tilted
    sorted_pairs = sorted(box_class_pairs, key=lambda pair: np.min(pair[0][:, 0]))

    # Unzip the sorted pairs back into separate lists
    sorted_boxes, sorted_classes = zip(*sorted_pairs) if box_class_pairs else ([], [])

    return list(sorted_boxes), list(sorted_classes)


def add_padding_to_match_size(small_image, target_size):
    """
    Resize and pad a smaller image to match a target size

    Args:
        small_image: The smaller input image (e.g., cropped meter)
        target_size: Tuple (width, height) of the target size

    Returns:
        Resized and padded image matching the target dimensions
    """
    # Get original dimensions
    small_h, small_w = small_image.shape[:2]
    target_w, target_h = target_size

    # Calculate how much to scale the small image to fit within target dimensions
    # while maintaining aspect ratio
    scale = min(target_w / small_w, target_h / small_h) * 0.8  # Use 80% of available space

    # Calculate new dimensions after scaling
    new_w = int(small_w * scale)
    new_h = int(small_h * scale)

    # Resize the small image
    resized_image = cv2.resize(small_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a blank image with target dimensions
    result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    result.fill(240)  # Light gray background

    # Calculate position to place the resized image (center it)
    pos_x = (target_w - new_w) // 2
    pos_y = (target_h - new_h) // 2

    # Place the resized image on the blank canvas
    result[pos_y:pos_y+new_h, pos_x:pos_x+new_w] = resized_image

    return result


def recognize_meter_reading(image_path, save_debug_images=False, debug_dir="debug_images"):
    """
    Process bulk flow meter image and recognize the digits

    Args:
        image_path: Path to the input image
        save_debug_images: Whether to save intermediate images for debugging
        debug_dir: Directory to save debug images

    Returns:
        Meter reading as a string
    """
    # Create directory for debug images if needed
    if save_debug_images:
        os.makedirs(debug_dir, exist_ok=True)

    # Step 1: Load the image
    image = cv2.imread(image_path)
    if image is None:
        return "Error: Could not load image"

    # Save original image if debugging
    if save_debug_images:
        original_filename = os.path.join(debug_dir, "1_original.jpg")
        cv2.imwrite(original_filename, image)

    # Step 3: Detect meter in the image (using global model)
    broader_results = broader_meter_model(image)

    # Check if meter detection results exist
    meter_detected = False
    straightened_meter = None

    # If debug mode, create a copy of the image for visualization
    debug_image = image.copy() if save_debug_images else None

    for result in broader_results:
        if hasattr(result, 'obb') and result.obb is not None:
            xyxyxyxy = result.obb.xyxyxyxy  # polygon format with 4-points

            # Draw bounding boxes on the debug image
            if save_debug_images:
                for i, box in enumerate(xyxyxyxy):
                    # Convert tensor to numpy array and reshape to 4 points format
                    points = box.cpu().numpy().reshape(-1, 2).astype(np.int32)
                    cv2.polylines(debug_image, [points], isClosed=True, color=(0, 0, 255), thickness=2)

                # Save the image with bounding boxes
                debug_bbox_filename = os.path.join(debug_dir, "2_meter_detection.jpg")
                cv2.imwrite(debug_bbox_filename, debug_image)

            # Process the first detected meter (assuming one meter per image)
            if len(xyxyxyxy) > 0:
                meter_detected = True
                # Convert tensor to numpy array and reshape to 4 points format
                meter_points = xyxyxyxy[0].cpu().numpy().reshape(-1, 2).astype(np.int32)

                # Step 4: Use perspective transform to straighten the meter
                straightened_meter = straighten_box(image, meter_points)

                if save_debug_images:
                    straightened_filename = os.path.join(debug_dir, "3_straightened_meter.jpg")
                    cv2.imwrite(straightened_filename, straightened_meter)

                # Step 5: Enhance the image for better digit recognition
                enhanced_meter = enhance_image(straightened_meter)

                if save_debug_images:
                    enhanced_filename = os.path.join(debug_dir, "4_enhanced_meter.jpg")
                    cv2.imwrite(enhanced_filename, enhanced_meter)

                break

    if not meter_detected or straightened_meter is None:
        return "Error: No meter detected in the image"


    # Add padding to the enhanced meter image
    original_size = (image.shape[1], image.shape[0])
    padded_meter = add_padding_to_match_size(enhanced_meter, original_size)

    # Save the padded meter image for debugging
    if save_debug_images:
        padded_filename = os.path.join(debug_dir, "4b_padded_meter.jpg")
        cv2.imwrite(padded_filename, padded_meter)

    # Step 6: Detect individual numbers on the enhanced meter image (using global model)
    digit_results = individual_numbers_model(image)

    digit_boxes = []
    digit_classes = []

    # Process digit detection results
    for result in digit_results:
        if hasattr(result, 'obb') and result.obb is not None:
            xyxyxyxy = result.obb.xyxyxyxy  # polygon format with 4-points
            class_ids = result.obb.cls.int()  # class ids of each box
            confidences = result.obb.conf  # confidence scores

            # Create a debug image for digits if needed
            if save_debug_images:
                digits_debug_image = padded_meter.copy()

            # Process each detected digit
            for i, box in enumerate(xyxyxyxy):
                # Convert tensor to numpy array and reshape to 4 points format
                points = box.cpu().numpy().reshape(-1, 2).astype(np.int32)
                class_id = class_ids[i].item()
                confidence = confidences[i].item()

                # Only consider high confidence detections
                if confidence > 0.0:
                    print(f"Detected digit {class_id} with confidence: {confidence:.3f}")
                    digit_boxes.append(points)
                    digit_classes.append(class_id)

                    # Draw digits on debug image
                    if save_debug_images:
                        # Draw the oriented bounding box
                        cv2.polylines(digits_debug_image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

                        # Add class label with confidence
                        label = f"{class_id} ({confidence:.2f})"
                        text_pos = (points[0][0], points[0][1] - 5)

                        font_scale = 0.5
                        font_thickness = 1

                        # Create a filled rectangle for text background
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                        cv2.rectangle(digits_debug_image,
                                    (text_pos[0], text_pos[1] - text_size[1] - 2),
                                    (text_pos[0] + text_size[0], text_pos[1] + 2),
                                    (0, 255, 0),
                                    -1)

                        # Put text on the background
                        cv2.putText(digits_debug_image, label, text_pos,
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

            # Save the digits debug image
            if save_debug_images and digit_boxes:
                digits_filename = os.path.join(debug_dir, "5_detected_digits.jpg")
                cv2.imwrite(digits_filename, digits_debug_image)

    # Step 7: Post-processing - sort the digits from left to right
    if not digit_boxes:
        return "Error: No digits detected on the meter"

    sorted_boxes, sorted_classes = sort_boxes_by_position(digit_boxes, digit_classes)

    # Draw sorted digits on a debug image if needed
    if save_debug_images and sorted_boxes:
        sorted_debug_image = padded_meter.copy()
        for i, (box, digit) in enumerate(zip(sorted_boxes, sorted_classes)):
            # Draw the oriented bounding box
            cv2.polylines(sorted_debug_image, [box], isClosed=True, color=(255, 0, 0), thickness=2)

            # Add class label with position index
            label = f"{digit} ({i})"
            text_pos = (box[0][0], box[0][1] - 5)

            font_scale = 0.5
            font_thickness = 1

            # Create a filled rectangle for text background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            cv2.rectangle(sorted_debug_image,
                        (text_pos[0], text_pos[1] - text_size[1] - 2),
                        (text_pos[0] + text_size[0], text_pos[1] + 2),
                        (255, 0, 0),
                        -1)

            # Put text on the background
            cv2.putText(sorted_debug_image, label, text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

        sorted_filename = os.path.join(debug_dir, "6_sorted_digits.jpg")
        cv2.imwrite(sorted_filename, sorted_debug_image)

    # Step 8: Extract the class labels and join them to form the digit sequence
    meter_reading = ''.join([str(cls) for cls in sorted_classes])

    return meter_reading


def process_image(image_path, save_debug=True):
    """
    Process a single image and display the meter reading

    Args:
        image_path: Path to the input image
        save_debug: Whether to save debug images
    """
    print(f"Processing image: {image_path}")

    # Create a debug directory using the image base name
    if save_debug:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        debug_dir = f"debug_{base_name}"
    else:
        debug_dir = "debug_images"

    # Recognize the meter reading
    meter_reading = recognize_meter_reading(image_path, save_debug, debug_dir)

    print(f"Detected meter reading: {meter_reading}")
    return meter_reading


def process_directory(directory_path, save_debug=False):
    """
    Process all images in a directory

    Args:
        directory_path: Path to the directory containing images
        save_debug: Whether to save debug images

    Returns:
        Dictionary mapping image names to meter readings
    """
    # Get all image files in the directory
    image_files = []
    for ext in ['*.jpeg', '*.jpg', '*.png']:
        image_files.extend(glob.glob(os.path.join(directory_path, ext)))

    print(f"Found {len(image_files)} images to process")

    results = {}
    for image_path in image_files:
        image_name = os.path.basename(image_path)
        meter_reading = process_image(image_path, save_debug)
        results[image_name] = meter_reading

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bulk Flow Meter Reading Recognition")
    parser.add_argument("--input", required=True, help="Path to input image or directory containing images")
    parser.add_argument("--debug", action="store_true", help="Save intermediate debug images")
    parser.add_argument("--output", help="Path to save results (JSON file for directory input)")

    args = parser.parse_args()

    if os.path.isfile(args.input):
        # Process a single image
        result = process_image(args.input, args.debug)

        # Save result to file if specified
        if args.output:
            with open(args.output, 'w') as f:
                f.write(f"Meter Reading: {result}\n")
            print(f"Result saved to {args.output}")

    elif os.path.isdir(args.input):
        # Process all images in the directory
        results = process_directory(args.input, args.debug)

        # Save results to JSON file
        import json
        output_file = args.output if args.output else "meter_readings.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"Results for {len(results)} images saved to {output_file}")

    else:
        print(f"Error: Input path {args.input} does not exist")
