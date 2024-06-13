import numpy as np
import cv2

def segment_image(image_path: str,
                  num_segments: int) -> list[np.ndarray]:
    image = cv2.imread(image_path)
    height, _, _ = image.shape
    
    segment_height = height // num_segments
    segments = []
    
    for i in range(num_segments):
        segment = image[i * segment_height:(i + 1) * segment_height, :]
        segments.append(segment)
    
    return segments

if __name__ == "__main__":
    image_path = "path_to_your_image.jpg"
    num_segments = 10
    segments = segment_image(image_path, num_segments)
    for i, segment in enumerate(segments):
        cv2.imwrite(f"segment_{i}.jpg", segment)
