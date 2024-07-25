import cv2
from pathlib import Path
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('best.pt')

def process_images(input_folder, output_folder):
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

        cv2.imwrite(str(output_path / img_path.name), img)
        print(f"Processed {img_path.name} with center at ({center_x}, {center_y})")

# Process images from all camera folders
# for i in range(1, 9):
#     process_images(f'cam{i}', f'output_cam{i}')

process_images('../../data/cam1/', '../../output/_cam1/')

print("Processing complete.")
