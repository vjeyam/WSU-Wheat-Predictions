from ultralytics import YOLO

# Load a pretrained YOLOv8x model
model = YOLO('yolo8x.pt') # Replace 'yolo8x.pt' with path to the 'best.pt'

# Run inference with arguments
result = model.predict('', save=True) # Save predictions to 'runs/detect/exp'

for results in result:
    boxes = results.boxes
    masks = result.masks
    keypoints = result.keypoints
    probs = result.probs
    obb = result.obb
    result.show()  # Show results
    result.save()  # Save results