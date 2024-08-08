import os
import cv2

def process_images(input_dir: str, output_dir: str) -> None:
    """
    Processes images in the input directory to extract contours from binary masks
    and convert them to polygons, which are then saved as text files in the output directory.

    Args:
        input_dir (str): Path to the directory containing input images (binary masks).
        output_dir (str): Path to the directory where the output polygon files will be saved.

    Returns:
        None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for j in os.listdir(input_dir):
        image_path = os.path.join(input_dir, j)
        
        # Load the binary mask
        mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print(f"Error: Could not read image {image_path}")
            continue
        
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        
        H, W = mask.shape
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Convert the contours to polygons
        polygons = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 200:
                polygon = []
                for point in cnt:
                    x, y = point[0]
                    polygon.append([x / W])
                    polygon.append([y / H])
                polygons.append(polygon)
        
        # Save the polygons
        output_path = os.path.join(output_dir, j)[:-4] + '.txt'
        with open(output_path, 'w') as f:
            for polygon in polygons:
                for p_, p in enumerate(polygon):
                    if p_ == len(polygon) - 1:
                        f.write('{}\n'.format(p))
                    elif p_ == 0:
                        f.write('0 {}'.format(p))
                    else:
                        f.write('{} '.format(p))
            print(f"Saved {output_path}")

if __name__ == "__main__":
    input_dir = 'datasets/wheat/train/cam_num/'
    output_dir = 'datasets/wheat/train/labels/'
    process_images(input_dir, output_dir)