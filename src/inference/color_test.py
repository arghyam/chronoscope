import os
import shutil
import xml.etree.ElementTree as ET

import cv2
import numpy as np

# Define paths
IMAGES_DIR = "dataset1/images"
ANNOTATIONS_XML = "dataset1/annotations.xml"
OUTPUT_DIR = "classification"

# Output subfolders for classified images
red_dir = os.path.join(OUTPUT_DIR, "red")
not_red_dir = os.path.join(OUTPUT_DIR, "not_red")
os.makedirs(red_dir, exist_ok=True)
os.makedirs(not_red_dir, exist_ok=True)

def get_last_three_digit_crops(image, image_name, annotation_root):
    """
    Extracts cropped regions of the last 3 digits in an image based on XML annotations.
    """
    digit_boxes = []

    # Search for the image entry in XML
    for image_tag in annotation_root.findall(".//image"):
        if image_tag.get("name") == image_name:
            for polygon in image_tag.findall(".//polygon"):
                label = polygon.get("label")

                # Only process polygons labeled with digits
                if label.isdigit():
                    points = polygon.get("points")
                    point_list = [tuple(map(float, pt.split(","))) for pt in points.split(";")]
                    digit_boxes.append((label, np.array(point_list, dtype=np.int32)))
            break

    if len(digit_boxes) < 3:
        return None  # Not enough digits for classification

    # Sort digit boxes from left to right
    digit_boxes.sort(key=lambda x: x[1][:, 0].min())

    crops = []
    for _, box in digit_boxes[-3:]:  # Take the last 3 digits
        x, y, w, h = cv2.boundingRect(box)
        crop = image[y:y+h, x:x+w]
        crops.append(crop)

    return crops  # [n-2, n-1, n]

def is_last_digit_color_different_hsv(digit_crops, threshold=0.85):
    """
    Returns True if the last digit's color differs from the others based on HSV analysis.
    """
    # Convert all digit crops to HSV color space
    hsv_digits = [cv2.cvtColor(crop, cv2.COLOR_BGR2HSV) for crop in digit_crops]

    # Combine pixels of the 2nd and 3rd last digits as reference
    ref_pixels = np.concatenate([hsv.reshape(-1, 3) for hsv in hsv_digits[:2]], axis=0)
    last_pixels = hsv_digits[2].reshape(-1, 3)

    # Define tolerance range (H, S, V)
    margin = np.array([12, 40, 40])

    # Compute HSV bounds from reference digits
    min_vals = np.maximum(np.min(ref_pixels, axis=0) - margin, [0, 0, 0])
    max_vals = np.minimum(np.max(ref_pixels, axis=0) + margin, [180, 255, 255])

    # Only compare Hue and Saturation channels
    last_pixels_hs = last_pixels[:, :2]
    min_vals_hs = min_vals[:2]
    max_vals_hs = max_vals[:2]

    # Count how many pixels in last digit are within the reference range
    in_range = np.all((last_pixels_hs >= min_vals_hs) & (last_pixels_hs <= max_vals_hs), axis=1)
    match_ratio = np.sum(in_range) / len(last_pixels)

    return match_ratio < threshold  # True = likely red, False = similar to others

def classify_images(images_dir, xml_path, output_dir):
    """
    Main driver function that loads images, applies red-check, and saves to respective folders.
    """
    # Load XML tree
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for file in os.listdir(images_dir):
        # Skip non-image files
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(images_dir, file)
        image = cv2.imread(image_path)
        if image is None:
            continue

        # Extract last 3 digit crops
        digit_crops = get_last_three_digit_crops(image, file, root)
        if digit_crops is None:
            continue

        # Classify based on color deviation of last digit
        is_red = is_last_digit_color_different_hsv(digit_crops)

        # Choose target directory
        target_dir = red_dir if is_red else not_red_dir

        # Copy image to appropriate folder
        shutil.copy(image_path, os.path.join(target_dir, file))

    print("Classification and copying completed.")

# Run the classification
classify_images(IMAGES_DIR, ANNOTATIONS_XML, OUTPUT_DIR)
