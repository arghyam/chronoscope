import cv2
import pandas as pd
from utils import classify_bfm_image
from utils import direct_recognize_meter_reading
from utils import extract_digit_image
from utils import is_last_digit_color_different_hsv
from utils import load_bfm_classification
from utils import load_individual_numbers_model

# Path to the CSV file
csv_path = "dataset/decimal_red_digits.csv"


def process_image(image_path):
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

            # Check if we have at least 3 digits for comparison
            if sorted_boxes and len(sorted_boxes) >= 3:
                # Extract the last 3 digit images
                last_three_boxes = sorted_boxes[-3:]
                last_three_digits = [extract_digit_image(image, box) for box in last_three_boxes]

                # Classify the color of the last digit by comparing with previous two
                color_result = is_last_digit_color_different_hsv(last_three_digits)

                return {
                    "quality_status": "good",
                    "meter_reading": meter_reading,
                    "last_digit": sorted_classes[-1] if sorted_classes else None,
                    "color": "Red" if color_result['is_red'] else "Black",
                    "confidence": float(color_result['confidence'])
                }
            elif sorted_boxes and len(sorted_boxes) > 0:
                # Not enough digits for comparison, fallback to just reporting detected digits
                return {
                    "quality_status": "good",
                    "meter_reading": meter_reading,
                    "last_digit": sorted_classes[-1] if sorted_classes else None,
                    "color": "Unknown (need at least 3 digits for color comparison)",
                    "confidence": 0.0
                }
            else:
                return {"error": "No digits detected in the image"}
        else:
            return {"quality_status": "bad", "error": "Image classified as bad quality"}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":

    csv_path = "dataset/last_digit_red_annotations.csv"

    df = pd.read_csv(csv_path)

    # Add new columns for results
    df['Quality_Status'] = ''
    df['Meter_Reading'] = ''
    df['Last_Digit'] = ''
    df['Last_Digit_Color'] = ''
    df['Color_Confidence'] = ''
    df['Error'] = ''

    # for index, row in df.iterrows():
    #     if row['Set'] == "Dataset1":
    #         image_path = f"dataset/data_cleaned/filtered_data/dataset1/{row['File_Name']+'.jpeg'}"
    #         result = process_image(image_path)
    #     elif row['Set'] == "Dataset2":
    #         image_path = f"dataset/data_cleaned/filtered_data/dataset2/{row['File_Name']+'.png'}"
    #         result = process_image(image_path)

    for index, row in df.iterrows():
        print(f"Processing image: {index} out of {len(df)}")

        image_path = row['image_name']

        result = process_image(image_path)

        # Update results in dataframe
        if 'error' in result:
            df.at[index, 'Error'] = result['error']
        else:
            df.at[index, 'Quality_Status'] = result.get('quality_status', '')
            df.at[index, 'Meter_Reading'] = result.get('meter_reading', '')
            df.at[index, 'Last_Digit'] = result.get('last_digit', '')
            df.at[index, 'Last_Digit_Color'] = result.get('color', '')
            df.at[index, 'Color_Confidence'] = result.get('confidence', '')

    # Add actual color based on is_last_digit_red column
    df['actual'] = df['is_last_digit_red'].apply(lambda x: 'Red' if x else 'Black')

    # Add prediction accuracy column comparing actual vs predicted color
    df['Prediction_Accuracy'] = df.apply(lambda x: 'Correct' if x['actual'] == x['Last_Digit_Color'] else 'Incorrect', axis=1)

    # Save updated results back to CSV
    df.to_csv(csv_path, index=False)
