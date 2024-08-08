from ultralytics import YOLO

model_name = 'yolo8x-seg'
dataset_name = ''

model = YOLO(f"{model_name}.pt")

# Load the model
model = YOLO('') # Replace '' with path to the 'best.pt' or custom model.pt

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps  # a list contains map50-95 of each category

# Segmentation
metrics.seg.map  # map50-95
metrics.seg.map50  # map50
metrics.seg.map75  # map75
metrics.seg.maps  # a list contains map50-95 of each category