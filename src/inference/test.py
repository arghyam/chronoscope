import os

import cv2
import numpy as np
from ultralytics import YOLO

# Create directory to save results if it doesn't exist
save_dir = "dataset/inference_data_check"
os.makedirs(save_dir, exist_ok=True)

model = YOLO("runs_indi_full_10k/obb/train/weights/best.pt")  # load a custom model

# Predict with the model
image_path = "dataset/training_data/yolo_indivisual/test/images/20240920025300_C2272151_F21369_M7884102.png"
results = model(image_path)  # predict on an image

# Load the original image
img = cv2.imread(image_path)

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

            # Add class name and confidence (commented out for now)
            # label = f"{names[i]} {confs[i]:.2f}"
            # cv2.putText(img, label, (points[0][0], points[0][1] - 15),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 200), 1)
    else:
        print("No OBB detection results found.")

# Save the image with bounding boxes
image_name = os.path.basename(image_path)
save_path = os.path.join(save_dir, f"bbox_{image_name}")
cv2.imwrite(save_path, img)
print(f"Image with bounding boxes saved to {save_path}")

print(results)
