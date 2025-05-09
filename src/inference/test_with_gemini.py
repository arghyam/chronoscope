import base64
import logging
import os

import cv2
import numpy as np
from dotenv import load_dotenv
from google.generativeai import configure
from google.generativeai import GenerativeModel
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load environment variables for Gemini API
load_dotenv()

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
    Enhance the image to improve readability for OCR

    Args:
        image: Input image (numpy array)

    Returns:
        Enhanced image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Apply unsharp masking for sharpening
    gaussian = cv2.GaussianBlur(gray, (0, 0), 3.0)
    sharpened = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)

    # Convert back to BGR for compatibility with the rest of the pipeline
    enhanced = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    return enhanced

def encode_image(image):
    """Encode the image to base64."""
    try:
        # Convert image to jpg format in memory
        _, buffer = cv2.imencode('.jpg', image)
        # Convert to base64
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def process_image_with_gemini(image, api_key):
    """Process an image using Gemini 2.0 Flash for OCR."""
    # Get base64 encoded image
    image_data = encode_image(image)
    if not image_data:
        return None

    # Configure Google AI
    configure(api_key=api_key)

    # Initialize Gemini model
    model = GenerativeModel('gemini-2.0-flash-exp')

    # Process with Gemini
    try:
        response = model.generate_content(
            contents=[
                "Extract the meter reading from this image. Return ONLY the complete numeric value without any separators, commas, spaces, or extra text. For example if you see '123,456' or '123 456' or '1','2','3','4','5','6' then return only '123456'.",
                {
                    "mime_type": "image/jpeg",
                    "data": image_data
                }
            ],
            generation_config={
                "temperature": 0.4,
                "max_output_tokens": 1024,
            }
        )

        # Clean response to ensure we only have digits
        result = response.text.strip()
        # Extract only digits from the response
        digits_only = ''.join(c for c in result if c.isdigit())
        return digits_only if digits_only else "No numbers found."

    except Exception as e:
        return f"Error processing image: {e}"

# Create directory to save results if it doesn't exist
save_dir = "data/test_data"
os.makedirs(save_dir, exist_ok=True)

# Create directory for straightened meter images
meters_dir = os.path.join(save_dir, "meters")
os.makedirs(meters_dir, exist_ok=True)

# Create directories for cropped and enhanced images
cropped_dir = os.path.join(save_dir, "cropped")
enhanced_dir = os.path.join(save_dir, "enhanced")
os.makedirs(cropped_dir, exist_ok=True)
os.makedirs(enhanced_dir, exist_ok=True)

model = YOLO("models/weights_yolov11_medium.pt")  # load a custom model

# Predict with the model
image_path = "data/cleaned_data/yolo_final_data/test/images/field_21.jpeg"
results = model(image_path)  # predict on an image

# Load the original image
img = cv2.imread(image_path)

# Get API key from environment variables
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Warning: GOOGLE_API_KEY not found in environment variables!")

ocr_results = []

# Access the results
for result in results:
    # Check if OBB detection results exist
    if hasattr(result, 'obb') and result.obb is not None:
        xyxyxyxy = result.obb.xyxyxyxy  # polygon format with 4-points
        names = [result.names[cls.item()] for cls in result.obb.cls.int()]  # class name of each box
        confs = result.obb.conf  # confidence score of each box

        # Draw bounding boxes on the image (for visualization only)
        for i, box in enumerate(xyxyxyxy):
            # Convert tensor to numpy array and reshape to 4 points format
            points = box.cpu().numpy().reshape(-1, 2).astype(np.int32)

            # Draw the oriented bounding box with lighter red color and thinner line
            cv2.polylines(img, [points], isClosed=True, color=(0, 0, 200), thickness=1)

            # Add class name and confidence to the main visualization image only
            label = f"{names[i]} {confs[i]:.2f}"
            # cv2.putText(img, label, (points[0][0], points[0][1] - 15),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 200), 1)

            # Use a clean copy of the original image for perspective transformation
            # This ensures no labels or bounding box lines appear in the cropped image
            straightened_meter = straighten_box(cv2.imread(image_path), points)

            # Save the straightened/cropped meter image
            cropped_filename = f"cropped_{i}_{os.path.basename(image_path)}"
            cropped_save_path = os.path.join(cropped_dir, cropped_filename)
            cv2.imwrite(cropped_save_path, straightened_meter)

            # Apply image enhancement to improve OCR accuracy
            enhanced_meter = enhance_image(straightened_meter)

            # Save the enhanced meter image
            enhanced_filename = f"enhanced_{i}_{os.path.basename(image_path)}"
            enhanced_save_path = os.path.join(enhanced_dir, enhanced_filename)
            cv2.imwrite(enhanced_save_path, enhanced_meter)

            # Save the original image with bounding box
            meter_filename = f"bbox_meter_{i}_{os.path.basename(image_path)}"
            meter_save_path = os.path.join(meters_dir, meter_filename)
            cv2.imwrite(meter_save_path, img)

            # Process with Gemini for OCR if API key is available
            if api_key:
                ocr_result = process_image_with_gemini(enhanced_meter, api_key)
                ocr_results.append({
                    'meter_id': i,
                    'confidence': float(confs[i]),
                    'ocr_text': ocr_result
                })
                print(f"Meter {i}: OCR Result: {ocr_result}")
    else:
        print("No OBB detection results found.")

# # Save the image with bounding boxes
image_name = os.path.basename(image_path)
# save_path = os.path.join(save_dir, f"bbox_{image_name}")
# cv2.imwrite(save_path, img)
# print(f"Image with bounding boxes saved to {save_path}")

# Save OCR results to file
if ocr_results:
    import json
    ocr_save_path = os.path.join(save_dir, f"ocr_results_{os.path.splitext(image_name)[0]}.json")
    with open(ocr_save_path, 'w') as f:
        json.dump(ocr_results, f, indent=4)
    print(f"OCR results saved to {ocr_save_path}")
