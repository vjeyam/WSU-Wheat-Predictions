import numpy as np
import cv2

def calculate_ndvi(nir: np.ndarray,
                   red: np.ndarray) -> np.ndarray:
    ndvi = (nir - red) / (nir + red)
    return ndvi

def process_image(image_path: str):
    # Assuming the multispectral image has NIR and Red bands in different channels
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # Split the image into NIR and Red channels (adapt as needed)
    nir = image[:, :, 0]
    red = image[:, :, 1]
    
    ndvi = calculate_ndvi(nir, red)
    
    return {
        'NDVI': ndvi
    }

if __name__ == "__main__":
    image_path = "path_to_your_multispectral_image.tiff"
    indices = process_image(image_path)
    print(indices)