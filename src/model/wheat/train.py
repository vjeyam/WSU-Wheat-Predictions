
from ultralytics import YOLO
# Edit here to use your datasets
model_name = 'yolov8x-seg'  
dataset_name = 'wheat'

# Load a model
#model = YOLO(f"{model_name}.yaml")  # build a new model from YAML
model = YOLO(f"{model_name}.pt")  # load a pretrained model (recommended for training)
#model = YOLO(f"{model_name}.yaml").load(f"{model_name}.pt")  # build from YAML and transfer weights

# Set the device configuration to cpu if you don't have a GPU
device = 'cpu'  # '0' or '0,1,2,3'

# # Train the model
results = model.train(data=f"datasets/{dataset_name}/data.yaml", epochs=200, batch=4, imgsz=640, device=device)