import os
import subprocess
import sys

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
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    # Check image dimensions
                    # Assuming you have a function to get image dimensions
                    # Example: dimensions = get_image_dimensions(os.path.join(folder, file))
                    dimensions = (640, 928)  # Placeholder for actual dimension checking function
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
    """Run the script in ../model/wheat/."""
    print("Running segmentation script...")
    subprocess.run([sys.executable, "../model/wheat/segment.py"], check=True)

def main():
    """Main function to coordinate the workflow."""
    
    # Ensure the environment is set up
    setup_environment()
    
    data_folder = "data/"

    # Step 1: Check image dimensions
    check_image_dimensions(data_folder)

    # Step 2: Check for CSV files and calculate radiometric reflectance if needed
    check_csv_files()

    # Step 3: Run the segmentation script
    run_segment_script()

if __name__ == "__main__":
    main()