import cv2
from pathlib import Path
from ultralytics import YOLO
import csv

# Load the YOLOv8 model
model = YOLO('best.pt')

# Process images from the input folder and save the processed images to the output folder
def process_images(input_folder, output_folder, centers_list):
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for img_path in input_path.glob('*.png'):
        img = cv2.imread(str(img_path))
        results = model(img)
        
        for result in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2 = map(int, result[:4])
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            centers_list.append((img_path.name, center_x, center_y, 5, 5))

        cv2.imwrite(str(output_path / img_path.name), img)
        print(f"Processed {img_path.name} with center at ({center_x}, {center_y})")

# Save center coordinates to CSV with additional columns 'Width' and 'Height'
def save_to_csv(centers_list, output_csv):
    # Save the center coordinates to a CSV file
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'Center_X', 'Center_Y', 'Width', 'Height'])
        writer.writerows(centers_list)
    print(f"Saved center coordinates to {output_csv}")

# Split the CSV into NIR and RGB based on Center_X value
def split_csv(input_csv, nir_csv, rgb_csv):
    nir_centers = []
    rgb_centers = []
    
    # Read the center coordinates from the input CSV file
    with open(input_csv, mode='r') as file:
        reader = csv.reader(file)
        headers = next(reader)
        for row in reader:
            center_x = int(row[1])
            if center_x < 640:
                nir_centers.append(row)
            else:
                # Adjust the center coordinates for the RGB images
                row[1] = str(center_x - 640)
                rgb_centers.append(row)
    
    # Save the NIR and RGB center coordinates to separate CSV files
    with open(nir_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(nir_centers)
    print(f"Saved NIR center coordinates to {nir_csv}")

    # Save the RGB center coordinates to a separate CSV file
    with open(rgb_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rgb_centers)
    print(f"Saved RGB center coordinates to {rgb_csv}")

# Process images from all camera folders and save centers to CSV
for i in range(1, 9):
    centers_list = []  # Reset centers_list for each new camera folder
    process_images(f'../../data/cam{i}', f'../../model_output/cam{i}', centers_list)
    
    # Save the center coordinates to a combined CSV file
    combined_csv = f'../../model_output/cam{i}.csv'
    save_to_csv(centers_list, combined_csv)
    
    # Split the combined CSV into NIR and RGB CSV files
    nir_csv = f'../../model_output/cam{i}_nir.csv'
    rgb_csv = f'../../model_output/cam{i}_rgb.csv'
    split_csv(combined_csv, nir_csv, rgb_csv)

print("Processing complete.")