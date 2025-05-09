from ultralytics import YOLO

# Load the model
model = YOLO('yolov8m-obb.pt')

# Train the model
results = model.train(
    task='obb',
    data="/content/drive/MyDrive/arghyam_project/yolo_final_data_indivisual_numbers/data.yaml",
    epochs=100,
    imgsz=640,
    batch=32,
    # Add any other training parameters you need
)

# Optional: save training results
# results.save('path/to/save/results')

# Optional: evaluate the model after training
# metrics = model.val()

#server train

# from ultralytics import YOLO

# # Load the model
# model = YOLO("yolo11m-obb.pt")
# # Train the model
# results = model.train(
#     task='obb',
#     data="/opt/dlami/nvme/chronoscope/yolo_broader_meter/data.yaml",
#     epochs=200,
#     imgsz=640,
#     batch=32,
#     augment=False,
#     fliplr=0.0,
#     flipud=0.0,
#     mosaic=0.0,
#     mixup=0.0,
#     scale=0.0
#     # Add any other training parameters you need
# )

# Optional: save training results
# results.save('path/to/save/results')

# Optional: evaluate the model after training
# metrics = model.val()
