import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple

def check_model_exists(model_path: str) -> bool:
    """Check if the YOLO model file exists."""
    return os.path.exists(model_path)

def load_model(model_path: str) -> YOLO:
    """Load the YOLO model."""
    if not check_model_exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = YOLO(model_path)
    return model

def segment_image(model: YOLO, image_path: str) -> Tuple[np.ndarray, List[List[Tuple[int, int]]]]:
    """Segment the image using the YOLO model and return the mask and polygon points."""
    # Load image
    image = cv2.imread(image_path)
    
    # Perform segmentation
    results = model(image)
    
    # Get the segmentation masks and polygon points
    masks = results.masks.numpy()
    polygons = results.pandas().xyxy[0].values
    
    # Create a mask image (assuming masks are binary)
    mask_image = np.zeros_like(image)
    for mask in masks:
        mask_image[mask > 0.5] = [0, 255, 0]  # Green mask
    
    # Convert bounding boxes to polygon points
    polygon_points = []
    for poly in polygons:
        x1, y1, x2, y2 = int(poly[0]), int(poly[1]), int(poly[2]), int(poly[3])
        polygon_points.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    
    return mask_image, polygon_points

def save_segmented_image(output_image_path: str, segmented_image: np.ndarray):
    """Save the segmented image to the output path."""
    cv2.imwrite(output_image_path, segmented_image)

def save_polygon_points(output_text_path: str, polygon_points: List[List[Tuple[int, int]]]):
    """Save the polygon points to a text file."""
    with open(output_text_path, 'w') as file:
        for points in polygon_points:
            if len(points) >= 4:
                file.write(f"Polygon with {len(points)} sides:\n")
                for (x, y) in points:
                    file.write(f"({x}, {y})\n")
                file.write("\n")

def main():
    """Main function to perform image segmentation."""
    if len(sys.argv) != 3:
        print("Usage: python segment.py <model_path> <input_image_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    input_image_path = sys.argv[2]
    
    if not check_model_exists(model_path):
        print(f"Model file not found: {model_path}")
        sys.exit(1)
    
    if not os.path.exists(input_image_path):
        print(f"Input image not found: {input_image_path}")
        sys.exit(1)
    
    # Load the model
    model = load_model(model_path)
    
    # Perform segmentation
    segmented_image, polygon_points = segment_image(model, input_image_path)
    
    # Define output paths
    output_image_path = os.path.splitext(input_image_path)[0] + '_segmented.png'
    output_text_path = os.path.splitext(input_image_path)[0] + '_polygon_points.txt'
    
    # Save the segmented image
    save_segmented_image(output_image_path, segmented_image)
    
    # Save the polygon points
    save_polygon_points(output_text_path, polygon_points)
    
    print(f"Segmented image saved to: {output_image_path}")
    print(f"Polygon points saved to: {output_text_path}")

if __name__ == "__main__":
    main()