import glob
import os

import cv2
import numpy as np
from ultralytics import YOLO

# Create directory to save results if it doesn't exist
save_dir = "dataset/inference_data_check/indi_1"
os.makedirs(save_dir, exist_ok=True)

# Load the model
model = YOLO("runs_indi_full_10k/obb/train/weights/best.pt")  # replace with your model path

# Define the directory containing test images
test_images_dir = "dataset/training_data/yolo_indivisual/test/images"  # replace with your test images directory

# Get all image files in the directory
image_files = glob.glob(os.path.join(test_images_dir, "*.jpeg"))
image_files.extend(glob.glob(os.path.join(test_images_dir, "*.jpg")))
image_files.extend(glob.glob(os.path.join(test_images_dir, "*.png")))

print(f"Found {len(image_files)} images to process")

# Define a color map for labels 0-9 (distinct colors for each label)
# color_map = {
#     0: (255, 0, 0)
# }
color_map = {
    0: (255, 0, 0),      # Blue
    1: (0, 255, 0),      # Green
    2: (0, 0, 255),      # Red
    3: (255, 255, 0),    # Cyan
    4: (255, 0, 255),    # Magenta
    5: (0, 255, 255),    # Yellow
    6: (128, 0, 0),      # Dark blue
    7: (0, 128, 0),      # Dark green
    8: (0, 0, 128),      # Dark red
    9: (128, 128, 0)     # Dark cyan
}

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
            class_ids = result.obb.cls.int()  # class ids of each box

            # Draw bounding boxes on the image
            for i, box in enumerate(xyxyxyxy):
                # Convert tensor to numpy array and reshape to 4 points format
                points = box.cpu().numpy().reshape(-1, 2).astype(np.int32)

                # Get class ID (0-9)
                class_id = class_ids[i].item()

                # Use red color for all labels (BGR format)
                color = (0, 0, 255)  # Red in BGR

                # Draw the oriented bounding box
                cv2.polylines(img, [points], isClosed=True, color=color, thickness=2)

                # Add class label with white text on red background like in 4_visualise_yolo_obb.py
                label = f"{class_id}"
                text_pos = (points[0][0], points[0][1] - 5)  # Moved closer to the box

                # Smaller font size (reduced from 0.8 to 0.5) and thinner text (from 2 to 1)
                font_scale = 0.5
                font_thickness = 1

                # Create a filled rectangle for the text background (like the red box in visualise_yolo_obb)
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                cv2.rectangle(img,
                             (text_pos[0], text_pos[1] - text_size[1] - 2),
                             (text_pos[0] + text_size[0], text_pos[1] + 2),
                             (0, 0, 255),
                             -1)  # Filled rectangle

                # Put text in white on the red background with smaller font
                cv2.putText(img, label, text_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        else:
            print(f"No OBB detection results found for {image_path}")

    # Save the image with bounding boxes
    image_name = os.path.basename(image_path)
    save_path = os.path.join(save_dir, f"bbox_{image_name}")
    cv2.imwrite(save_path, img)
    print(f"Image with bounding boxes saved to {save_path}")

print("All images processed successfully!")
