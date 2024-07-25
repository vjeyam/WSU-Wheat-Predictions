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
            cv2.circle(img, (center_x, center_y), 5, (0, 255, 0), -1)
            cv2.putText(img, f'({center_x}, {center_y})', (center_x + 10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            centers_list.append((img_path.name, center_x, center_y))

        cv2.imwrite(str(output_path / img_path.name), img)
        print(f"Processed {img_path.name} with center at ({center_x}, {center_y})")

# Crop and resize images from 1280 x 928 to 640 x 928
def crop_images(input_folder):
    input_path = Path(input_folder)
    
    for img_path in input_path.glob('*.png'):
        img = cv2.imread(str(img_path))
        if img.shape[1] != 1280 or img.shape[0] != 928:
            print(f"Skipping {img_path.name} as it does not have the expected dimensions of 1280x928.")
            continue
        
        # Crop the right side to get 640 x 928
        cropped_img = img[:, :640]

        # Resize the cropped image to 640 x 928 (although it should already be the correct size)
        resized_img = cv2.resize(cropped_img, (640, 928))

        # Replace the original image with the resized image
        cv2.imwrite(str(img_path), resized_img)
        print(f"Cropped and resized {img_path.name}")

# Crop and resize images from all camera folders
def save_to_csv(centers_list, output_csv):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'Center_X', 'Center_Y'])
        writer.writerows(centers_list)
    print(f"Saved center coordinates to {output_csv}")

# Process images from all camera folders and save centers to CSV
centers_list = []
for i in range(1, 9):
    process_images(f'../../data/cam{i}', f'../../model_output/cam{i}', centers_list)
    crop_images(f'../../model_output/cam{i}')

save_to_csv(centers_list, '../../model_output/centers.csv')

print("Processing complete.")