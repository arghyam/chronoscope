import os
import csv
import cv2
import numpy as np
import pytesseract
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

def detect_orientation_using_ocr(image_path):
    """Detect image orientation using OCR to find text in different orientations."""
    try:
        # Read image with PIL to preserve color information
        pil_img = Image.open(image_path)
        
        # Create 4 rotated versions of the image (0, 90, 180, 270 degrees)
        rotations = [0, 90, 180, 270]
        confidence_scores = []
        
        for angle in rotations:
            # Rotate image
            rotated_img = pil_img.rotate(angle, expand=True)
            
            # Convert to OpenCV format for processing
            opencv_img = np.array(rotated_img)
            if len(opencv_img.shape) == 3:  # Color image
                opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_RGB2BGR)
            
            # Get OCR data with confidence values
            ocr_data = pytesseract.image_to_data(opencv_img, output_type=pytesseract.Output.DICT)
            
            # Calculate confidence score for this orientation
            # We'll use both the number of detected text blocks and their confidence
            confidences = [float(conf) for conf in ocr_data['conf'] if conf != '-1']
            
            if confidences:
                # Score is a combination of number of text blocks and their average confidence
                avg_conf = sum(confidences) / len(confidences)
                text_blocks = len([x for x in confidences if x > 50])  # Count blocks with confidence > 50%
                score = avg_conf * text_blocks
            else:
                score = 0
                
            confidence_scores.append(score)
        
        # Find the rotation with the highest confidence score
        best_orientation = rotations[np.argmax(confidence_scores)]
        
        # If the best orientation isn't 0 degrees, the image is rotated
        return best_orientation != 0
        
    except Exception as e:
        print(f"Error processing {image_path} with OCR: {e}")
        return None

def is_image_rotated(image_path):
    """Determine if an image is rotated using multiple methods."""
    # First check EXIF data (most reliable when available)
    exif_result = check_exif_orientation(image_path)
    if exif_result is not None:
        return exif_result
    
    # Use OCR-based orientation detection
    ocr_result = detect_orientation_using_ocr(image_path)
    if ocr_result is not None:
        return ocr_result
    
    # If all methods fail, assume image is properly oriented
    return False

def main():
    parser = argparse.ArgumentParser(description='Check image orientation using OCR and create a CSV report.')
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
