import os
import time

import cv2
import nest_asyncio
import streamlit as st
# Add these imports at the top

# Add this before your Streamlit code
nest_asyncio.apply()


from src.final_inference.inference_utils import (classify_bfm_image, direct_recognize_meter_reading ,
                                                 load_bfm_classification, load_individual_numbers_model,
                                                 extract_digit_image , is_last_digit_color_different_hsv)

# Set page configuration
st.set_page_config(
    page_title="Bulk Flow Meter Reading Recognition",
    layout="wide"
)

# Load models at app startup using st.cache_resource to ensure they're loaded only once
@st.cache_resource
def load_models():
    return {
        'bfm_classification': load_bfm_classification(),
        'individual_numbers': load_individual_numbers_model(),
    }

# Load models at startup
models = load_models()

# Create a beautiful heading
st.title("🔍 Bulk Flow Meter Reading Recognition")
st.markdown("---")

# File uploader for the image
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# Function to process the image
def process_uploaded_image(uploaded_file, models):
    # Create a temporary file to save the uploaded image
    temp_file_path = "temp_upload.jpg"

    # Convert uploaded file to image
    image_bytes = uploaded_file.getvalue()
    with open(temp_file_path, "wb") as f:
        f.write(image_bytes)

    # Start timer for inference time measurement
    start_time = time.time()

    try:
        # First, classify the image as good or bad
        classification_result = classify_bfm_image(temp_file_path, model=models['bfm_classification'])

        # Only proceed with digit recognition if image is classified as "good"
        if classification_result['prediction'].lower() == 'good':
            meter_reading, sorted_boxes, sorted_classes = direct_recognize_meter_reading(
                temp_file_path,
                models['individual_numbers']
            )

            print(meter_reading)
            # Check if we have at least 3 digits for color comparison
            if sorted_boxes and len(sorted_boxes) >= 3:
                # Read the image
                image = cv2.imread(temp_file_path)
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

                quality_status = "good"
            else:
                formatted_reading = "Need at least 3 digits for color detection"
                quality_status = "bad"
                color_result = {"is_red": False, "confidence": 0.0}
        else:
            formatted_reading = "Image classified as bad quality - recognition skipped"
            quality_status = "bad"
            color_result = {"is_red": False, "confidence": 0.0}

        # Load the image for display
        image = cv2.imread(temp_file_path)

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        # Calculate inference time at the end of all processing
        inference_time = time.time() - start_time

    return formatted_reading, inference_time, image, classification_result, quality_status, color_result

# Create columns for better layout
col1, col2 = st.columns([1, 1])

# Add a recognition button
if uploaded_file is not None:
    with col1:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    recognize_button = st.button("Recognize Meter Reading")

    if recognize_button:
        with st.spinner("Processing image..."):
            # Process the image with pre-loaded models
            meter_reading, inference_time, original_image, classification_result, quality_status, color_result = process_uploaded_image(uploaded_file, models)

            # Display results
            st.success(f"Processing complete! Time taken: {inference_time:.2f} seconds")

            # Display classification result
            st.markdown(f"### Image Quality: {quality_status.upper()}")

            # Display recognition result with larger font
            if quality_status == "good":
                st.markdown(f"### Recognized number: {meter_reading}")
                st.markdown(f"Last digit color: {'RED' if color_result['is_red'] else 'BLACK'} (Confidence: {color_result['confidence']:.2f})")
            else:
                st.error(meter_reading)

else:
    # Show instructions when no file is uploaded
    st.info("Please upload an image of a bulk flow meter to recognize its reading.")
    st.markdown("""
    ### Instructions:
    1. Use the file uploader above to select an image
    2. Click the 'Recognize Meter Reading' button
    3. Wait for the processing to complete
    4. View the image quality classification and recognized meter reading (if applicable)
    """)
