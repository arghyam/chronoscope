import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from classify_color import extract_digit_color
from classify_color import process_meter_reading_with_decimal
from classify_color import visualize_color_analysis

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def test_on_image(image_path, output_dir="results"):
    """
    Test color classification on a single image

    Args:
        image_path: Path to the image
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing image: {image_path}")

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    # Mock a list of digit images (for testing without having actual segmentation)
    # In a real scenario, these would come from the digit detection model
    # For this test, we'll just create dummy images with different colors

    # We'll simulate cropped digit images by splitting the image horizontally into sections
    height, width = image.shape[:2]
    num_segments = 5  # Simulating 5 digits
    segment_width = width // num_segments

    # Create a list of simulated digit images
    digit_images = []
    for i in range(num_segments):
        start_x = i * segment_width
        end_x = start_x + segment_width
        digit_image = image[:, start_x:end_x].copy()
        digit_images.append(digit_image)

    # Simulate digit classes (for testing)
    digit_classes = ['1', '2', '3', '4', '5']

    # Process with decimal detection
    result = process_meter_reading_with_decimal(digit_images, digit_classes)

    # Create visualization
    vis_path = os.path.join(output_dir, f"color_analysis_{Path(image_path).stem}.jpg")
    visualize_color_analysis(digit_images, vis_path)

    # Print results
    print(f"Reading: {result['reading']}")
    print(f"Has decimal: {result['has_decimal']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Saved visualization to: {vis_path}")
    print("-" * 50)

    return result

def test_custom_digit_colors():
    """
    Test with manually created digit images of different colors
    """
    output_dir = "results/custom_tests"
    os.makedirs(output_dir, exist_ok=True)

    # Create test images with different colors
    colors = [
        (0, 0, 0),        # Black
        (255, 0, 0),      # Pure Red
        (180, 0, 0),      # Dark Red
        (255, 100, 100),  # Light Red
        (100, 0, 0),      # Very Dark Red
        (50, 50, 50),     # Dark Gray
        (128, 0, 0),      # Maroon
        (200, 0, 0)       # Bright Red
    ]

    color_names = [
        "black", "pure_red", "dark_red", "light_red",
        "very_dark_red", "dark_gray", "maroon", "bright_red"
    ]

    # Create a test digit image for each color
    size = (100, 100)
    results = []

    for color, name in zip(colors, color_names):
        # Create a blank image
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        img.fill(240)  # Light gray background

        # Draw a digit with the test color
        digit_color = (color[2], color[1], color[0])  # BGR for OpenCV
        cv2.putText(img, "5", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, digit_color, 3)

        # Save the test image
        test_img_path = os.path.join(output_dir, f"test_digit_{name}.jpg")
        cv2.imwrite(test_img_path, img)

        # Analyze color
        color_info = extract_digit_color(img)

        # Create visualization
        vis_path = os.path.join(output_dir, f"color_analysis_{name}.jpg")
        digit_images = [img]  # Just one digit for testing
        visualize_color_analysis(digit_images, vis_path)

        # Print results
        print(f"Color: {name}")
        print(f"Is Red: {color_info['is_red']}")
        print(f"Confidence: {color_info['confidence']:.2f}")
        print(f"RGB: {color_info['dominant_rgb']}")
        print(f"Hue: {color_info['hue_value']:.1f}")
        print("-" * 50)

        results.append({
            'color_name': name,
            'is_red': color_info['is_red'],
            'confidence': color_info['confidence'],
            'rgb': color_info['dominant_rgb'],
            'hue': color_info['hue_value']
        })

    return results

def test_on_directory(directory_path, output_dir="results"):
    """
    Test color classification on all images in a directory

    Args:
        directory_path: Path to directory containing images
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all image files in the directory
    image_files = []
    for ext in ['jpg', 'jpeg', 'png']:
        image_files.extend(glob.glob(os.path.join(directory_path, f"*.{ext}")))

    # Process each image
    results = []
    for image_path in image_files:
        result = test_on_image(image_path, output_dir)
        results.append({
            'image': os.path.basename(image_path),
            'reading': result['reading'],
            'has_decimal': result['has_decimal'],
            'confidence': result['confidence']
        })

    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test the color classification for meter readings")
    parser.add_argument("--image", help="Path to a single image to process")
    parser.add_argument("--directory", help="Path to a directory containing images to process")
    parser.add_argument("--output", default="results", help="Directory to save output visualizations")
    parser.add_argument("--custom-test", action="store_true", help="Run tests with custom colored digits")

    args = parser.parse_args()

    if args.custom_test:
        test_custom_digit_colors()
    elif args.image:
        test_on_image(args.image, args.output)
    elif args.directory:
        test_on_directory(args.directory, args.output)
    else:
        print("Please provide either --image, --directory or --custom-test")
        print("Example: python test_color_classification.py --image path/to/image.jpg")
        print("Example: python test_color_classification.py --directory path/to/images/")
        print("Example: python test_color_classification.py --custom-test")
