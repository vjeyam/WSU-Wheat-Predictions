import os
import torch
from ultralytics import YOLO

model_name = 'yolov8x'
dataset_name = 'wheat'

# Build the path to the data.yaml file relative to the script directory
data_path = os.path.join(os.path.dirname(__file__), f"datasets/{dataset_name}/data.yaml")
train_images_path = os.path.join(os.path.dirname(__file__), f"datasets/{dataset_name}/train/images")
val_images_path = os.path.join(os.path.dirname(__file__), f"datasets/{dataset_name}/val/images")

# Print paths for debugging
print(f"Using data.yaml at: {data_path}")
print(f"Train images path: {train_images_path}")
print(f"Validation images path: {val_images_path}")

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load a model
model = YOLO(f"{model_name}.pt")

# Train the model
results = model.train(data=data_path,
                      epochs=100,
                      batch=4,
                      imgsz=640,
                      device=device)