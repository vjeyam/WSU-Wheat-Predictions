from ultralytics import YOLO
from typing import Any

model_name: str = 'yolo8x'
dataset_name: str = ''

model: YOLO = YOLO(f"{model_name}.pt")

# Load the model
model = YOLO('')  # Replace '' with the path to the 'best.pt' or custom model.pt

# Validate the model
metrics: Any = model.val()  # No arguments needed, dataset and settings remembered

# Bounding box metrics
metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps  # A list containing map50-95 of each category

# Segmentation metrics
metrics.seg.map  # map50-95
metrics.seg.map50  # map50
metrics.seg.map75  # map75
metrics.seg.maps  # A list containing map50-95 of each category