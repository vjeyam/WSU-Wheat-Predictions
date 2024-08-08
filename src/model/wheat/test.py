from ultralytics import YOLO
from typing import List, Optional

# Load a pretrained YOLOv8x model
model: YOLO = YOLO('yolo8x.pt')  # Replace 'yolo8x.pt' with the path to 'best.pt'

# Run inference with arguments
result = model.predict('', save=True)  # Save predictions to 'runs/detect/exp'

for results in result:
    boxes: Optional[List] = results.boxes  # List of detected bounding boxes
    masks: Optional[List] = result.masks  # Segmentation masks (if available)
    keypoints: Optional[List] = result.keypoints  # Keypoints detected in the image (if available)
    probs: Optional[List] = result.probs  # Class probabilities (if available)
    obb: Optional[List] = result.obb  # Oriented bounding boxes (if available)
    
    results.show()  # Show results
    results.save()  # Save results