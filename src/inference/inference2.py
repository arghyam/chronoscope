import glob
import os

import cv2
import numpy as np
from ultralytics import YOLO

# Load only the individual numbers model
individual_numbers_model = YOLO("runs_indi_full_10k/obb/train/weights/best.pt")

def enhance_image(image):
    """
    Enhance the image to improve readability for OCR while preserving color
    """
    # Convert to HSV color space to separate brightness from color
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Apply adaptive thresholding to the value channel (brightness)
    thresh = cv2.adaptiveThreshold(
        v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Apply unsharp masking for sharpening to the value channel
    gaussian = cv2.GaussianBlur(v, (0, 0), 3.0)
    sharpened_v = cv2.addWeighted(v, 1.5, gaussian, -0.5, 0)

    # Increase contrast in the value channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_v = clahe.apply(sharpened_v)

    # Merge channels back with enhanced value channel
    enhanced_hsv = cv2.merge([h, s, enhanced_v])

    # Convert back to BGR color space
    enhanced = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

    # Apply color boost to make digits stand out more
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=10)

    return enhanced

def sort_boxes_by_position(boxes, classes):
    """
    Sort detected digit boxes by their x-coordinate for left-to-right reading order.
    """
    # Create a list of (box, class) tuples
    box_class_pairs = list(zip(boxes, classes))

    # For oriented bounding boxes, use the leftmost x-coordinate of each box
    sorted_pairs = sorted(box_class_pairs, key=lambda pair: np.min(pair[0][:, 0]))

    # Unzip the sorted pairs back into separate lists
    sorted_boxes, sorted_classes = zip(*sorted_pairs) if box_class_pairs else ([], [])

    return list(sorted_boxes), list(sorted_classes)

def calculate_iou_improved(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two polygon boxes more accurately
    using OpenCV's contour operations

    Args:
        box1: First box as a numpy array of 4 points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        box2: Second box as a numpy array of 4 points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]

    Returns:
        IoU value between 0 and 1
    """
    # Convert boxes to contour format
    box1_contour = box1.reshape(-1, 1, 2).astype(np.int32)
    box2_contour = box2.reshape(-1, 1, 2).astype(np.int32)

    # Find bounding rectangle for both boxes to determine optimal canvas size
    x1, y1, w1, h1 = cv2.boundingRect(box1_contour)
    x2, y2, w2, h2 = cv2.boundingRect(box2_contour)

    # Create canvas that fits both boxes with some padding
    canvas_width = max(x1 + w1, x2 + w2) + 10
    canvas_height = max(y1 + h1, y2 + h2) + 10

    # Create masks for each box
    mask1 = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    mask2 = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

    # Draw filled polygons
    cv2.fillPoly(mask1, [box1_contour], 1)
    cv2.fillPoly(mask2, [box2_contour], 1)

    # Calculate area of each box
    area1 = cv2.countNonZero(mask1)
    area2 = cv2.countNonZero(mask2)

    # Calculate intersection
    intersection = cv2.bitwise_and(mask1, mask2)
    intersection_area = cv2.countNonZero(intersection)

    # Calculate union
    union_area = area1 + area2 - intersection_area

    # Return IoU
    if union_area == 0:
        return 0
    return intersection_area / union_area

def remove_overlapping_boxes_improved(boxes, classes, confidences, iou_threshold=0.3, distance_threshold=30):
    """
    Remove overlapping boxes using both IoU and center distance criteria

    Args:
        boxes: List of bounding boxes
        classes: List of class labels
        confidences: List of confidence scores
        iou_threshold: IoU threshold above which boxes are considered overlapping
        distance_threshold: Distance threshold below which box centers are considered too close

    Returns:
        Filtered boxes, classes, and confidences
    """
    if len(boxes) == 0:
        return [], [], []

    # Combine data for sorting
    box_data = list(zip(boxes, classes, confidences))

    # Sort by confidence (descending)
    box_data.sort(key=lambda x: x[2], reverse=True)

    # Initialize lists for keeping filtered boxes
    filtered_boxes = []
    filtered_classes = []
    filtered_confidences = []

    # Process boxes in order of confidence
    while box_data:
        # Get the box with the highest confidence
        current_box, current_class, current_conf = box_data.pop(0)

        # Calculate center of current box
        current_center = np.mean(current_box, axis=0)

        # Add to filtered list
        filtered_boxes.append(current_box)
        filtered_classes.append(current_class)
        filtered_confidences.append(current_conf)

        # Check remaining boxes
        remaining_boxes = []
        for box, cls, conf in box_data:
            # Calculate center of this box
            box_center = np.mean(box, axis=0)

            # Calculate distance between centers
            center_distance = np.linalg.norm(current_center - box_center)

            # Calculate IoU
            iou = calculate_iou_improved(current_box, box)

            # Keep box if it's far enough AND has low IoU
            if center_distance > distance_threshold and iou < iou_threshold:
                remaining_boxes.append((box, cls, conf))
            else:
                # Print debug info about removed box
                print(f"Removed box: class={cls}, conf={conf:.2f}, IoU={iou:.2f}, distance={center_distance:.2f}")

        # Update box_data with remaining boxes
        box_data = remaining_boxes

    return filtered_boxes, filtered_classes, filtered_confidences

def direct_recognize_meter_reading(image_path, save_debug_images=False, debug_dir="debug_images"):
    """
    Process image and directly recognize digits without meter detection

    Args:
        image_path: Path to the input image
        save_debug_images: Whether to save intermediate images for debugging
        debug_dir: Directory to save debug images

    Returns:
        Meter reading as a string
    """
    # Define color map for digits 0-9
    color_map = {
        0: (255, 0, 0),      # Blue
        1: (0, 255, 0),      # Green
        2: (0, 0, 255),      # Red
        3: (255, 255, 0),    # Cyan
        4: (255, 0, 255),    # Magenta
        5: (0, 255, 255),    # Yellow
        6: (128, 0, 0),      # Dark blue
        7: (0, 128, 0),      # Dark green
        8: (0, 0, 128),      # Dark red
        9: (128, 128, 0)     # Dark cyan
    }

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

    # Step 2: Enhance the image for better digit recognition
    enhanced_image = enhance_image(image)

    if save_debug_images:
        enhanced_filename = os.path.join(debug_dir, "2_enhanced_image.jpg")
        cv2.imwrite(enhanced_filename, enhanced_image)

    # Step 3: Detect individual numbers directly on the enhanced image
    digit_results = individual_numbers_model(enhanced_image)

    digit_boxes = []
    digit_classes = []
    digit_confidences = []

    # Process digit detection results
    for result in digit_results:
        if hasattr(result, 'obb') and result.obb is not None:
            xyxyxyxy = result.obb.xyxyxyxy  # polygon format with 4-points
            class_ids = result.obb.cls.int()  # class ids of each box
            confidences = result.obb.conf  # confidence scores

            # Create a debug image with all detections if needed
            if save_debug_images:
                all_detections_image = enhanced_image.copy()

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
                    digit_confidences.append(confidence)

                    # Draw on the all detections debug image
                    if save_debug_images:
                        bbox_color = color_map[class_id]
                        cv2.polylines(all_detections_image, [points], isClosed=True, color=bbox_color, thickness=2)
                        label = f"{class_id} ({confidence:.2f})"
                        text_pos = (points[0][0], points[0][1] - 5)
                        font_scale = 0.5
                        font_thickness = 1
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                        cv2.rectangle(all_detections_image,
                                    (text_pos[0], text_pos[1] - text_size[1] - 2),
                                    (text_pos[0] + text_size[0], text_pos[1] + 2),
                                    bbox_color,
                                    -1)
                        cv2.putText(all_detections_image, label, text_pos,
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

            # Save all detections debug image
            if save_debug_images and digit_boxes:
                all_detections_filename = os.path.join(debug_dir, "3_all_detections.jpg")
                cv2.imwrite(all_detections_filename, all_detections_image)

    # Step 4: Remove overlapping boxes using improved method
    if digit_boxes:
        print(f"Before filtering: {len(digit_boxes)} detections")
        digit_boxes, digit_classes, digit_confidences = remove_overlapping_boxes_improved(
            digit_boxes, digit_classes, digit_confidences, iou_threshold=0.3, distance_threshold=30
        )
        print(f"After filtering: {len(digit_boxes)} detections")

    # Create debug image for filtered digits if needed
    if save_debug_images and digit_boxes:
        filtered_debug_image = enhanced_image.copy()

        # Draw each filtered digit
        for i, (box, digit, conf) in enumerate(zip(digit_boxes, digit_classes, digit_confidences)):
            # Get color for this digit
            bbox_color = color_map[digit]

            # Draw the oriented bounding box
            cv2.polylines(filtered_debug_image, [box], isClosed=True, color=bbox_color, thickness=2)

            # Add class label with confidence
            label = f"{digit} ({conf:.2f})"
            text_pos = (box[0][0], box[0][1] - 5)

            font_scale = 0.5
            font_thickness = 1

            # Create a filled rectangle for text background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            cv2.rectangle(filtered_debug_image,
                        (text_pos[0], text_pos[1] - text_size[1] - 2),
                        (text_pos[0] + text_size[0], text_pos[1] + 2),
                        bbox_color,
                        -1)

            # Put text on the background
            cv2.putText(filtered_debug_image, label, text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

        filtered_filename = os.path.join(debug_dir, "4_filtered_digits.jpg")
        cv2.imwrite(filtered_filename, filtered_debug_image)

    # Step 5: Post-processing - sort the digits from left to right
    if not digit_boxes:
        return "Error: No digits detected in the image"

    sorted_boxes, sorted_classes = sort_boxes_by_position(digit_boxes, digit_classes)

    # Draw sorted digits on a debug image if needed
    if save_debug_images and sorted_boxes:
        sorted_debug_image = enhanced_image.copy()
        for i, (box, digit) in enumerate(zip(sorted_boxes, sorted_classes)):
            # Get color for this digit
            bbox_color = color_map[digit]

            # Draw the oriented bounding box
            cv2.polylines(sorted_debug_image, [box], isClosed=True, color=bbox_color, thickness=2)

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
                        bbox_color,
                        -1)

            # Put text on the background
            cv2.putText(sorted_debug_image, label, text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

        sorted_filename = os.path.join(debug_dir, "5_sorted_digits.jpg")
        cv2.imwrite(sorted_filename, sorted_debug_image)

    # Step 6: Extract the class labels and join them to form the digit sequence
    meter_reading = ''.join([str(cls) for cls in sorted_classes])

    return meter_reading

def process_image(image_path, save_debug=True):
    """
    Process a single image and display the meter reading
    """
    print(f"Processing image: {image_path}")

    # Create a debug directory using the image base name
    if save_debug:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        debug_dir = f"debug_{base_name}"
    else:
        debug_dir = "debug_images"

    # Recognize the meter reading directly
    meter_reading = direct_recognize_meter_reading(image_path, save_debug, debug_dir)

    print(f"Detected meter reading: {meter_reading}")
    return meter_reading

def process_directory(directory_path, save_debug=False):
    """
    Process all images in a directory
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
