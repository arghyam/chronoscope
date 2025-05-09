import glob
import os

import cv2
import numpy as np
from ultralytics import YOLO

# Create directory to save results if it doesn't exist
save_dir = "data/test_data_medium"
os.makedirs(save_dir, exist_ok=True)

# Load the model
model = YOLO("models/weights_yolov11_medium.pt")  # load a custom model

# Define the directory containing test images
test_images_dir = "data/cleaned_data/yolo_final_data/test/images"

# Get all image files in the directory
image_files = glob.glob(os.path.join(test_images_dir, "*.jpeg"))
image_files.extend(glob.glob(os.path.join(test_images_dir, "*.jpg")))
image_files.extend(glob.glob(os.path.join(test_images_dir, "*.png")))

print(f"Found {len(image_files)} images to process")

# Process each image
for image_path in image_files:
    print(f"Processing {image_path}...")

    # Predict with the model
    results = model(image_path)  # predict on an image

    # Load the original image
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error loading image: {image_path}")
        continue

    # Access the results
    for result in results:
        # Check if OBB detection results exist
        if hasattr(result, 'obb') and result.obb is not None:
            xyxyxyxy = result.obb.xyxyxyxy  # polygon format with 4-points
            names = [result.names[cls.item()] for cls in result.obb.cls.int()]  # class name of each box
            confs = result.obb.conf  # confidence score of each box

            # Draw bounding boxes on the image
            for i, box in enumerate(xyxyxyxy):
                # Convert tensor to numpy array and reshape to 4 points format
                points = box.cpu().numpy().reshape(-1, 2).astype(np.int32)

                # Draw the oriented bounding box with lighter red color and thinner line
                cv2.polylines(img, [points], isClosed=True, color=(0, 0, 200), thickness=1)

                # Add class name and confidence (uncomment if needed)
                # label = f"{names[i]} {confs[i]:.2f}"
                # cv2.putText(img, label, (points[0][0], points[0][1] - 15),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 200), 1)
        else:
            print(f"No OBB detection results found for {image_path}")

    # Save the image with bounding boxes
    image_name = os.path.basename(image_path)
    save_path = os.path.join(save_dir, f"bbox_{image_name}")
    cv2.imwrite(save_path, img)
    print(f"Image with bounding boxes saved to {save_path}")

print("All images processed successfully!")
