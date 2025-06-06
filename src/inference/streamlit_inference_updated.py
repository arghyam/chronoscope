import os
import time

import cv2
import streamlit as st
from inference1 import direct_recognize_meter_reading
from utils import classify_bfm_image
from utils import load_bfm_classification
from utils import load_individual_numbers_model

# Set page configuration
st.set_page_config(
    page_title="Bulk Flow Meter Reading Recognition",
    layout="wide"
)

# Create a beautiful heading
st.title("🔍 Bulk Flow Meter Reading Recognition")
st.markdown("---")

# File uploader for the image
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# Function to process the image
def process_uploaded_image(uploaded_file):
    # Create a temporary file to save the uploaded image
    temp_file_path = "temp_upload.jpg"

    # Convert uploaded file to image
    image_bytes = uploaded_file.getvalue()
    with open(temp_file_path, "wb") as f:
        f.write(image_bytes)

    # Start timer for inference time measurement
    start_time = time.time()

    # Load models
    bfm_classification_model = load_bfm_classification()
    individual_numbers_model = load_individual_numbers_model()

    # First, classify the image as good or bad
    classification_result = classify_bfm_image(temp_file_path, model=bfm_classification_model)

    # Only proceed with digit recognition if image is classified as "good"
    if classification_result['prediction'].lower() == 'good':
        meter_reading,_ ,_  = direct_recognize_meter_reading(temp_file_path, individual_numbers_model)
        quality_status = "good"
    else:
        meter_reading = "Image classified as bad quality - recognition skipped"
        quality_status = "bad"

    # Calculate inference time
    inference_time = time.time() - start_time

    # Load the image for display
    image = cv2.imread(temp_file_path)

    # Clean up the temporary file
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

    return meter_reading, inference_time, image, classification_result, quality_status

# Create columns for better layout
col1, col2 = st.columns([1, 1])

# Add a recognition button
if uploaded_file is not None:
    with col1:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    recognize_button = st.button("Recognize Meter Reading")

    if recognize_button:
        with st.spinner("Processing image..."):
            # Process the image
            meter_reading, inference_time, original_image, classification_result, quality_status = process_uploaded_image(uploaded_file)

            # Display results
            st.success(f"Processing complete! Time taken: {inference_time:.2f} seconds")

            # Display classification result
            st.markdown(f"### Image Quality: {quality_status.upper()}")
            st.markdown(f"Confidence: {classification_result['confidence']:.2f}")

            # Display recognition result with larger font
            if quality_status == "good":
                st.markdown(f"### Recognized number: {meter_reading}")
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
