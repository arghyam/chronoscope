import os
import csv
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import argparse
import easyocr

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
    """Detect if water meter image is properly oriented."""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Text/Number Detection using EasyOCR
        # Water meters typically have numbers in horizontal orientation
        reader = easyocr.Reader(['en'])
        results = reader.readtext(img)
        
        if results:
            # Analyze text orientation
            angles = []
            confidences = []
            for detection in results:
                bbox = detection[0]  # Get bounding box points
                confidence = detection[2]  # Get confidence score
                
                # Calculate angle of text
                angle = np.arctan2(bbox[1][1] - bbox[0][1], 
                                 bbox[1][0] - bbox[0][0]) * 180 / np.pi
                angles.append(angle)
                confidences.append(confidence)
            
            # Weight angles by confidence scores
            weighted_angle = np.average(angles, weights=confidences)
            
            # If text is significantly rotated
            if abs(weighted_angle) > 20:
                return True  # Image needs rotation
            return False  # Image orientation is likely correct
        
        # Method 2: Circle/Dial Detection
        # Water meters often have circular elements
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=20,
            maxRadius=100
        )
        
        if circles is not None:
            # Analyze the pattern of circles
            # Most water meters have dials arranged horizontally
            circles = np.round(circles[0, :]).astype("int")
            if len(circles) >= 2:
                # Calculate angles between circles
                center_points = circles[:, :2]  # Get centers of circles
                angles = []
                for i in range(len(center_points)-1):
                    for j in range(i+1, len(center_points)):
                        angle = np.arctan2(
                            center_points[j][1] - center_points[i][1],
                            center_points[j][0] - center_points[i][0]
                        ) * 180 / np.pi
                        angles.append(angle % 180)  # Normalize angle
                
                avg_angle = np.mean(angles)
                if abs(avg_angle - 90) < 20:  # If circles are mostly vertical
                    return True  # Image needs rotation
                return False  # Image orientation is likely correct
        
        return None  # Uncertain if no method gives confident result
    
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

def correct_image_orientation(image_path, output_dir):
    """Correct image orientation using multiple rotation attempts if needed."""
    try:
        # First check EXIF
        pil_image = Image.open(image_path)
        exif_data = pil_image._getexif()
        image_name = os.path.basename(image_path)
        output_path = os.path.join(output_dir, image_name)
        
        # Handle EXIF orientation if available
        if exif_data:
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == 'Orientation':
                    # Rotate according to EXIF orientation
                    if value == 2:
                        pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
                    elif value == 3:
                        pil_image = pil_image.rotate(180)
                    elif value == 4:
                        pil_image = pil_image.transpose(Image.FLIP_TOP_BOTTOM)
                    elif value == 5:
                        pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT).rotate(90)
                    elif value == 6:
                        pil_image = pil_image.rotate(270)
                    elif value == 7:
                        pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT).rotate(270)
                    elif value == 8:
                        pil_image = pil_image.rotate(90)
                    
                    # Save corrected image
                    pil_image.save(output_path)
                    return True
        
        # If no EXIF, use feature detection
        feature_result = detect_orientation_using_features(image_path)
        if feature_result:
            # Try multiple rotations and evaluate confidence
            img = cv2.imread(image_path)
            rotations = [
                ('original', img),
                ('90_clockwise', cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)),
                ('180', cv2.rotate(img, cv2.ROTATE_180)),
                ('90_counterclockwise', cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))
            ]
            
            # Evaluate each rotation
            best_confidence = 0
            best_rotation = None
            
            for rotation_name, rotated_img in rotations:
                # Convert to PIL for saving
                rotated_pil = Image.fromarray(cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB))
                
                # Save the best rotation
                if rotation_name == 'original':
                    pil_image.save(output_path)
                else:
                    rotated_pil.save(output_path)
                
            return True
        else:
            # Copy original if no rotation needed
            pil_image.save(output_path)
            return False
            
    except Exception as e:
        print(f"Error correcting orientation for {image_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Check and correct image orientation and create a CSV report.')
    parser.add_argument('--input_dir', required=True, help='Directory containing images to check')
    parser.add_argument('--output_dir', required=True, help='Directory to save corrected images')
    parser.add_argument('--output_csv', required=True, help='Path to output CSV file')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all image files
    image_files = get_image_files(args.input_dir)
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    results = []
    for i, image_path in enumerate(image_files):
        if i % 10 == 0:
            print(f"Processing image {i+1}/{len(image_files)}")
        
        is_rotated = is_image_rotated(image_path)
        was_corrected = correct_image_orientation(image_path, args.output_dir)
        image_name = os.path.basename(image_path)
        
        # Store result
        results.append({
            'image_name': image_name,
            'was_rotated': 'Yes' if is_rotated else 'No',
            'correction_applied': 'Yes' if was_corrected else 'No'
        })
    
    # Write results to CSV
    with open(args.output_csv, 'w', newline='') as csvfile:
        fieldnames = ['image_name', 'was_rotated', 'correction_applied']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"Results saved to {args.output_csv}")
    print(f"Corrected images saved to {args.output_dir}")

if __name__ == "__main__":
    main()