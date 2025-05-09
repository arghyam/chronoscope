import os

import cv2
import pandas as pd

from src.inference.utils import classify_bfm_image
from src.inference.utils import direct_recognize_meter_reading
from src.inference.utils import extract_digit_image
from src.inference.utils import load_bfm_classification
from src.inference.utils import load_individual_numbers_model

# Path to the CSV file
csv_path = "dataset/last_digit_red_annotations.csv"

# Ensure output directory exists
output_dir_red = "dataset/data_cleaned/color_classification/red"
output_dir_black = "dataset/data_cleaned/color_classification/black"
os.makedirs(output_dir_red, exist_ok=True)
os.makedirs(output_dir_black, exist_ok=True)


def process_image(image_path , output_dir):
    """
    Process a single image to detect and classify the last digit color

    Args:
        image_path: Path to the image file

    Returns:
        Dictionary with classification results
    """
    try:
        # Load the image using OpenCV
        image = cv2.imread(image_path)

        if image is None:
            return {"error": f"Could not load image at {image_path}"}

        # Load models (if not already loaded)
        bfm_classification_model = load_bfm_classification()
        individual_numbers_model = load_individual_numbers_model()

        # First, classify the image as good or bad
        classification_result = classify_bfm_image(image_path, model=bfm_classification_model)

        # Only proceed with digit recognition if image is classified as "good"
        if classification_result['prediction'].lower() == 'good':
            meter_reading, sorted_boxes, sorted_classes = direct_recognize_meter_reading(
                image_path, individual_numbers_model)

            # Check if we have at least one digit detected
            if sorted_boxes and len(sorted_boxes) >= 1:
                # Extract the last digit image
                last_box = sorted_boxes[-1]
                last_digit_image = extract_digit_image(image, last_box)

                # Get the image filename without path
                image_filename = os.path.basename(image_path)

                # Save the cropped image to the specified folder
                output_path = os.path.join(output_dir, image_filename)
                cv2.imwrite(output_path, last_digit_image)

                return {
                    "status": "success",
                    "meter_reading": meter_reading,
                    "last_digit": sorted_classes[-1],
                    "saved_to": output_path
                }
            else:
                return {"status": "error", "message": "No digits detected in the image"}
        else:
            return {"status": "error", "message": "Image classified as bad quality"}

    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    # Load the CSV file with image paths
    df = pd.read_csv(csv_path)

    # Process each image
    results = []
    for index, row in df.iterrows():
        image_path = row['image_name']  # Adjust column name if different
        if row["is_last_digit_red"] == False :
            output_dir = output_dir_black
        else:
            output_dir = output_dir_red
        print(f"Processing image {index+1}/{len(df)}: {image_path}")
        result = process_image(image_path, output_dir)
        results.append(result)

    # Summarize results
    success_count = sum(1 for r in results if r.get('status') == 'success')
    print(f"Processed {len(results)} images. Successfully extracted {success_count} last digits.")
