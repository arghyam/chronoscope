import os

import pandas as pd
from inference import recognize_meter_reading
from tqdm import tqdm

def process_images_and_compare():
    # Read the digit sequences CSV with dtype specification to keep leading zeros
    df = pd.read_csv('dataset/inference_data_check/test_images/digit_sequences.csv', dtype={'digit_sequence': str})

    # Base path for images
    base_image_path = 'dataset/training_data/yolo_indivisual/test/images'

    # Prepare results list
    results = []

    # Process each image
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        image_name = row['image_name']
        ground_truth = row['digit_sequence']

        # Construct full image path
        image_path = os.path.join(base_image_path, image_name)

        # Skip if image doesn't exist
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_name} not found in {base_image_path}")
            continue

        try:
            # Perform inference
            predicted_sequence = recognize_meter_reading(image_path, save_debug_images=False)

            # Add to results
            results.append({
                'image_name': image_name,
                'ground_truth': ground_truth,
                'predicted': predicted_sequence,
                'match': ground_truth == predicted_sequence
            })

        except Exception as e:
            print(f"Error processing {image_name}: {str(e)}")
            results.append({
                'image_name': image_name,
                'ground_truth': ground_truth,
                'predicted': "ERROR",
                'match': False
            })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Calculate accuracy
    accuracy = (results_df['match'].sum() / len(results_df)) * 100

    # Save results to CSV
    output_file = 'dataset/inference_data_check/test_images/inference_results.csv'
    results_df.to_csv(output_file, index=False)

    print(f"\nResults saved to {output_file}")
    print(f"Overall accuracy: {accuracy:.2f}%")

    return results_df

if __name__ == "__main__":
    results_df = process_images_and_compare()
