import os
import subprocess
import sys
from src.scripts import process_rgb_images, process_nir_images
from model.panel import process_images
from src.scripts.radiometric_correction import process_cameras  # Import process_cameras from radiometric_correction

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
                    dimensions = get_image_dimensions(os.path.join(folder, file))
                    if dimensions != (640, 928):
                        resize_needed = True
                        break
            if resize_needed:
                break

    if resize_needed:
        print("Images need resizing. Calling split_cam_images.py...")
        subprocess.run([sys.executable, "src/scripts/split_cam_images.py"], check=True)
    else:
        print("All images are already 640x928.")

def get_image_dimensions(image_path):
    """Get the dimensions of an image."""
    from PIL import Image
    with Image.open(image_path) as img:
        return img.size

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
        process_cameras(base_path='data/')  # Call process_cameras to handle radiometric correction

def run_segment_script():
    """Run the segmentation script in ../model/wheat/."""
    model_path = "../model/wheat/best.pt"
    image_folder = "data/corrected_images/"  # Folder with radiometrically corrected images
    output_folder = "data/segmented_images/"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        sys.exit(1)
    
    # Run segmentation on all images in the corrected images folder
    for file in os.listdir(image_folder):
        if file.endswith(('.png')):
            input_image_path = os.path.join(image_folder, file)
            output_image_path = os.path.join(output_folder, os.path.splitext(file)[0] + '_segmented.png')
            output_text_path = os.path.join(output_folder, os.path.splitext(file)[0] + '_polygon_points.txt')
            
            # Call process_rgb_images and process_nir_images if needed
            process_rgb_images(input_image_path, output_image_path)
            process_nir_images(input_image_path, output_image_path)
            
            # Run segment.py script
            result = subprocess.run([sys.executable, "../model/wheat/segment.py", model_path, input_image_path],
                                    capture_output=True, text=True)
            
            # Check if segmentation was successful
            if result.returncode != 0:
                print(f"Error during segmentation of {input_image_path}:\n", result.stderr)
                continue
            
            # Save polygon points to text file
            with open(output_text_path, 'w') as file:
                file.write(result.stdout)
            
            print(f"Segmentation completed for {input_image_path}")
            print(f"Segmented image saved to: {output_image_path}")
            print(f"Polygon points saved to: {output_text_path}")

def main():
    """Main function to coordinate the workflow."""
    
    # Ensure the environment is set up
    setup_environment()
    
    data_folder = "data/"

    # Step 1: Check image dimensions
    check_image_dimensions(data_folder)

    # Process RGB and NIR images
    process_images(data_folder, "data/corrected_images/")
    
    # Process cameras (handles radiometric correction)
    process_cameras(base_path=data_folder)

    # Step 2: Check for CSV files and calculate radiometric reflectance if needed
    check_csv_files()

    # Step 3: Run the segmentation script
    run_segment_script()

if __name__ == "__main__":
    main()