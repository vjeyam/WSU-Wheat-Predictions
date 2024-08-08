from ultralytics import YOLO
from typing import Any

# Edit here to use your datasets
model_name: str = 'yolov8x-seg'
dataset_name: str = 'wheat'

# Load a model
# model: YOLO = YOLO(f"{model_name}.yaml")  # Build a new model from YAML
model: YOLO = YOLO(f"{model_name}.pt")  # Load a pretrained model (recommended for training)
# model: YOLO = YOLO(f"{model_name}.yaml").load(f"{model_name}.pt")  # Build from YAML and transfer weights

# Set the device configuration to CPU if you don't have a GPU
device: str = 'cpu'  # '0' or '0,1,2,3' for GPU

# Train the model
results: Any = model.train(data=f"datasets/{dataset_name}/data.yaml", 
                           epochs=200, 
                           batch=4, 
                           imgsz=640, 
                           device=device)