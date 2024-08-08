from ultralytics import YOLO
from typing import List

# Load a pretrained YOLOv8x model
model: YOLO = YOLO('yolo8x.pt')  # Replace 'yolo8x.pt' with the path to 'best.pt'

# Run inference with arguments
result = model.predict('', save=True)  # Save predictions to 'runs/detect/exp'

for results in result:
    boxes = results.boxes  # List of detected bounding boxes
    masks = result.masks  # Segmentation masks
    keypoints = result.keypoints  # Keypoints detected in the image
    probs = result.probs  # Class probabilities
    obb = result.obb  # Oriented bounding boxes (if available)
    results.show()  # Show results
    results.save()  # Save results