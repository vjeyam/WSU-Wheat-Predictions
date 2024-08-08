import os
import subprocess
import sys
from src.scripts import process_rgb_images, process_nir_images, process_cameras
from model.panel import process_images, save_to_csv, split_csv
from PIL import Image
from typing import Tuple

def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """
    Get the dimensions of an image.

    Args:
        image_path (str): Path to the image file.

    Returns:
        Tuple[int, int]: The width and height of the image.
    """
    with Image.open(image_path) as img:
        return img.size

def setup_environment():
    """Ensure that the virtual environment is set up and dependencies are installed."""
    setup_script_path = os.path.join(os.getcwd(), "src", "scripts", "setup_env.py")
    
    # Run the setup_env.py script
    result = subprocess.run([sys.executable, setup_script_path], capture_output=True, text=True)
    
    # Check the result
    if result.returncode != 0:
        print("Error setting up the environment:\n", result.stderr)
        sys.exit(1)
    
    print("Environment setup successfully")

def check_image_dimensions(data_folder):
    """Check if all images in cam folders are 640x928, and call split_cam_images.py if not."""
    cam_folders = [os.path.join(data_folder, f"cam{i}") for i in range(1, 9)]
    resize_needed = False

    for folder in cam_folders:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                if file.endswith(('.png')):
                    image_path = os.path.join(folder, file)
                    # Check image dimensions
                    dimensions = get_image_dimensions(image_path)
                    if dimensions != (640, 928):
                        resize_needed = True
                        break
            if resize_needed:
                break

    if resize_needed:
        print("Images need resizing. Calling split_cam_images.py...")
        subprocess.run([sys.executable, "src/scripts/split_cam_images.py"], check=True)
        # Process RGB and NIR images after resizing
        process_rgb_images(data_folder, "data/output_rgb")
        process_nir_images(data_folder, "data/output_nir")
    else:
        print("All images are already 640x928.")
        # Process RGB and NIR images if no resizing was needed
        process_rgb_images(data_folder, "data/output_rgb")
        process_nir_images(data_folder, "data/output_nir")

def check_csv_files():
    """Check for the presence of CSV files and call detect.py if necessary."""
    csv_folder = "data/csv/"
    required_csv_files = {f"cam{i}.csv" for i in range(1, 9)}

    existing_csv_files = {f for f in os.listdir(csv_folder) if f.endswith(".csv")}
    missing_csv_files = required_csv_files - existing_csv_files

    if missing_csv_files:
        print("CSV files are missing. Running detect.py to calculate radiometric reflectance...")
        subprocess.run([sys.executable, "../model/panel/detect.py"], check=True)
    else:
        print("All required CSV files are present. Proceeding with radiometric correction...")
        subprocess.run([sys.executable, "src/scripts/radiometric_correction.py"], check=True)

def run_segment_script():
    """Run the script in ../model/wheat/segment.py."""
    print("Running segmentation script...")
    subprocess.run([sys.executable, "../model/wheat/segment.py"], check=True)

def main():
    """Main function to coordinate the workflow."""
    
    # Ensure the environment is set up
    setup_environment()
    
    data_folder = "../../data/"

    # Step 1: Check image dimensions and process images
    check_image_dimensions(data_folder)

    # Step 2: Check for CSV files and calculate radiometric reflectance if needed
    check_csv_files()

    # Step 3: Process cameras
    process_cameras()

    # Example usage of YOLO-based processing and CSV handling
    centers_list = []
    process_images(data_folder, "data/output_rgb", centers_list)
    save_to_csv(centers_list, 'data/centers.csv')

    # Step 4: Run the segmentation script
    run_segment_script()

if __name__ == "__main__":
    main()