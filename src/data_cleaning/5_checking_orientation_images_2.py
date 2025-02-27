import os
import csv
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import argparse

def get_image_files(directory):
    """Get all image files from a directory."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    return image_files

def check_exif_orientation(image_path):
    """Check if image has orientation information in EXIF data."""
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        
        if exif_data is None:
            return None
        
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == 'Orientation':
                # 1 means normal orientation, other values indicate rotation
                return value != 1
        
        return None  # No orientation tag found
    except Exception as e:
        print(f"Error reading EXIF data for {image_path}: {e}")
        return None

def detect_orientation_using_features(image_path):
    """Detect if image is properly oriented using feature detection."""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Use text detection as a heuristic (text is usually horizontal)
        # This is a simple approach and may not work for all images
        # More sophisticated methods could be implemented
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Use Hough Line Transform to detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is None:
            return None
        
        # Count horizontal and vertical lines
        horizontal_count = 0
        vertical_count = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Classify as horizontal or vertical with some tolerance
            if angle < 20 or angle > 160:
                horizontal_count += 1
            elif 70 < angle < 110:
                vertical_count += 1
        
        # If there are significantly more horizontal lines, image is likely properly oriented
        if horizontal_count > vertical_count * 1.5:
            return False  # Not rotated
        elif vertical_count > horizontal_count * 1.5:
            return True  # Possibly rotated
        
        return None  # Uncertain
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def is_image_rotated(image_path):
    """Determine if an image is rotated using multiple methods."""
    # First check EXIF data
    exif_result = check_exif_orientation(image_path)
    if exif_result is not None:
        return exif_result
    
    # If no EXIF data, try feature-based detection
    feature_result = detect_orientation_using_features(image_path)
    if feature_result is not None:
        return feature_result
    
    # If all methods fail, assume image is properly oriented
    return False

def main():
    parser = argparse.ArgumentParser(description='Check image orientation and create a CSV report.')
    parser.add_argument('--input_dir', required=True, help='Directory containing images to check')
    parser.add_argument('--output_csv', required=True, help='Path to output CSV file')
    args = parser.parse_args()
    
    # Get all image files
    image_files = get_image_files(args.input_dir)
    print(f"Found {len(image_files)} images to process")
    
    # Check orientation for each image
    results = []
    for i, image_path in enumerate(image_files):
        if i % 10 == 0:
            print(f"Processing image {i+1}/{len(image_files)}")
        
        is_rotated = is_image_rotated(image_path)
        image_name = os.path.basename(image_path)
        
        # Store result
        results.append({
            'image_name': image_name,
            'is_properly_oriented': 'No' if is_rotated else 'Yes'
        })
    
    # Write results to CSV
    with open(args.output_csv, 'w', newline='') as csvfile:
        fieldnames = ['image_name', 'is_properly_oriented']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"Results saved to {args.output_csv}")

if __name__ == "__main__":
    main()