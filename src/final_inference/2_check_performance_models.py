import os

import cv2
import pandas as pd
import yaml

from src.final_inference.inference_utils import classify_bfm_image
from src.final_inference.inference_utils import classify_color_image
from src.final_inference.inference_utils import direct_recognize_meter_reading
from src.final_inference.inference_utils import extract_digit_image
from src.final_inference.inference_utils import load_bfm_classification
from src.final_inference.inference_utils import load_color_classification_model
from src.final_inference.inference_utils import load_individual_numbers_model

def load_config():
    config_path = os.path.join('src', 'final_inference', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_models(config):
    return {
        'bfm_classification': load_bfm_classification(model_path=config['models']['bfm_classification']),
        'individual_numbers': load_individual_numbers_model(model_path=config['models']['individual_numbers']),
        'color_classification': load_color_classification_model(model_path=config['models']['color_classification'])
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

            # Extract and classify the last digit's color if digits were detected
            if sorted_boxes and len(sorted_boxes) > 0:
                # Read the image
                image = cv2.imread(image_path)
                # Extract the last digit image
                last_box = sorted_boxes[-1]
                last_digit_image = extract_digit_image(image, last_box)

                # Classify the color of the last digit
                color_result = classify_color_image(
                    last_digit_image,
                    model=models['color_classification']
                )

                # Convert meter reading to float and adjust based on color
                try:
                    # Preserve the original string length
                    original_length = len(meter_reading)
                    numeric_reading = float(meter_reading)

                    if color_result['prediction'].lower() == 'red':
                        numeric_reading = numeric_reading / 10
                        formatted_reading = f"{numeric_reading:.1f}".zfill(original_length + 2)
                    else:
                        formatted_reading = str(int(numeric_reading)).zfill(original_length)
                except ValueError:
                    formatted_reading = meter_reading
            else:
                formatted_reading = "no_digits"
                color_result = {"prediction": "unknown", "confidence": 0.0}
        else:
            formatted_reading = "bad_image"
            color_result = {"prediction": "unknown", "confidence": 0.0}

        return formatted_reading, classification_result['prediction'], color_result['prediction']

    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return "error", "error", "unknown"

def evaluate_models():
    # Load config
    config = load_config()

    # Load models
    print("Loading models...")
    models = load_models(config)

    # Read the golden dataset CSV using path from config
    df = pd.read_csv(config['dataset']['golden_dataset']['csv_files']['digit_sequences'])

    # Create lists to store results
    results = []

    print("Processing images...")
    # Process each image
    for idx, row in df.iterrows():
        print(f"Processing image {idx + 1} of {len(df)}")
        image_name = row['image_name']
        true_sequence = row['digit_sequence']
        true_color = row['has_decimal']
        is_missing = row['is_missing']

        image_path = os.path.join(config['dataset']['golden_dataset']['images'], image_name)

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

    # Save results using path from config
    output_path = config['dataset']['golden_dataset']['csv_files']['evaluation_results']
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    evaluate_models()
