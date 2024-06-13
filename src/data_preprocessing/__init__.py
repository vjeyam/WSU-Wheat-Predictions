"""
data_preprocessing package

This package contains modules for preprocessing images in the wheat project. 
It includes functionality for extracting vegetation indices, segmenting images, 
and correcting images based on a reference panel.

Modules:
    extract_vegetation_indices: Functions for calculating vegetation indices such as NDVI from multispectral images.
    segment_images: Functions for segmenting images into multiple segments.
    correct_images: Functions for correcting images based on a reference panel.

Example usage:
    from src.data_preprocessing import calculate_ndvi, segment_image, correct_image

    # Calculate NDVI from NIR and Red channels
    ndvi = calculate_ndvi(nir_image, red_image)

    # Segment an image into 10 parts
    segments = segment_image("path_to_image.jpg", 10)

    # Correct an image using a reference panel
    corrected_image = correct_image("path_to_image.jpg", "path_to_reference_panel.jpg")
"""

from .extract_vegetation_indices import calculate_ndvi, process_image
from .segment_images import segment_image
from .correct_images import correct_image