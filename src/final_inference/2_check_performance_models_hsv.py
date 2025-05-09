import os

import cv2
import pandas as pd

from src.final_inference.inference_utils import classify_bfm_image
from src.final_inference.inference_utils import direct_recognize_meter_reading
from src.final_inference.inference_utils import extract_digit_image
from src.final_inference.inference_utils import is_last_digit_color_different_hsv
from src.final_inference.inference_utils import load_bfm_classification
from src.final_inference.inference_utils import load_individual_numbers_model

def load_models():
    return {
        'bfm_classification': load_bfm_classification(),
        'individual_numbers': load_individual_numbers_model(),
    }

def compare_readings(true_sequence, recognized_reading):
    """
    Compare true sequence with recognized reading
    Returns: 'correct', 'wrong', or 'no_output'
    """
    if recognized_reading in ['bad_image', 'no_digits', 'error']:
        return 'no_output'

    try:
        # Convert both to float for comparison
        true_val = float(true_sequence)
        rec_val = float(recognized_reading)

        # Compare with small tolerance for floating point numbers
        if abs(true_val - rec_val) < 1e-10:
            return 'correct'
        return 'wrong'
    except ValueError:
        return 'no_output'

def process_image(image_path, models):
    try:
        # First, classify the image as good or bad
        classification_result = classify_bfm_image(image_path, model=models['bfm_classification'])

        # Only proceed with digit recognition if image is classified as "good"
        if classification_result['prediction'].lower() == 'good':
            meter_reading, sorted_boxes, sorted_classes = direct_recognize_meter_reading(
                image_path,
                models['individual_numbers']
            )

            # Check if we have at least 3 digits for color comparison
            if sorted_boxes and len(sorted_boxes) >= 3:
                # Read the image
                image = cv2.imread(image_path)
                # Extract the last three digit images
                last_three_boxes = sorted_boxes[-3:]
                last_three_digits = [extract_digit_image(image, box) for box in last_three_boxes]

                # Classify the color of the last digit by comparing with previous two
                color_result = is_last_digit_color_different_hsv(last_three_digits)

                # Convert meter reading to float and adjust based on color
                try:
                    # Preserve the original string length
                    original_length = len(meter_reading)
                    numeric_reading = float(meter_reading)

                    if color_result['is_red']:
                        numeric_reading = numeric_reading / 10
                        formatted_reading = f"{numeric_reading:.1f}".zfill(original_length + 2)
                    else:
                        formatted_reading = str(int(numeric_reading)).zfill(original_length)
                except ValueError:
                    formatted_reading = meter_reading
            else:
                formatted_reading = "no_digits"
                color_result = {"is_red": False, "confidence": 0.0}
        else:
            formatted_reading = "bad_image"
            color_result = {"is_red": False, "confidence": 0.0}

        return formatted_reading, classification_result['prediction'], 'red' if color_result['is_red'] else 'black'

    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return "error", "error", "unknown"

def evaluate_models():
    # Load models
    print("Loading models...")
    models = load_models()

    # Read the golden dataset CSV
    df = pd.read_csv('dataset/golden_dataset/csv_files/digit_sequences.csv')

    # Create lists to store results
    results = []

    print("Processing images...")
    # Process each image
    for idx, row in df.iterrows():
        print(f"Processing image {idx} of {len(df)}")
        image_name = row['image_name']
        true_sequence = row['digit_sequence']
        true_color = row['has_decimal']
        is_missing = row['is_missing']

        image_path = os.path.join('dataset/golden_dataset/images', image_name)

        if os.path.exists(image_path):
            recognized_reading, quality, color = process_image(image_path, models)

            # Compare readings
            comparison_result = compare_readings(true_sequence, recognized_reading)

            results.append({
                'image_name': image_name,
                'true_sequence': true_sequence,
                'true_color': true_color,
                'is_missing': is_missing,
                'recognized_reading': recognized_reading,
                'quality_prediction': quality,
                'color_prediction': color,
                'comparison_result': comparison_result
            })

    # Create DataFrame and save results
    results_df = pd.DataFrame(results)

    # Calculate and print statistics
    total_images = len(results_df)
    correct_count = len(results_df[results_df['comparison_result'] == 'correct'])
    wrong_count = len(results_df[results_df['comparison_result'] == 'wrong'])
    no_output_count = len(results_df[results_df['comparison_result'] == 'no_output'])

    print("\nEvaluation Statistics:")
    print(f"Total Images: {total_images}")
    print(f"Correct Readings: {correct_count} ({(correct_count/total_images)*100:.2f}%)")
    print(f"Wrong Readings: {wrong_count} ({(wrong_count/total_images)*100:.2f}%)")
    print(f"No Output: {no_output_count} ({(no_output_count/total_images)*100:.2f}%)")

    # Save results
    output_path = 'dataset/golden_dataset/csv_files/model_evaluation_results_hsv.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    evaluate_models()
