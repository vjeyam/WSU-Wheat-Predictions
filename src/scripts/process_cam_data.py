import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2

def plot_center_values(output_folder='../assets'):
    for i in range(1, 9):
        # Plot for cam{i}_nir.csv
        nir_df = pd.read_csv(f'../model_output/cam{i}_nir.csv')
        plt.figure(figsize=(10, 7))
        plt.scatter(nir_df['Center_X'], nir_df['Center_Y'], c='r', marker='o')
        plt.title(f'Camera {i} NIR Center_X and Center_Y values')
        plt.xlabel('Center_X')
        plt.ylabel('Center_Y')
        plt.grid(True)
        plt.savefig(f'{output_folder}/cam{i}_nir_center_x_y.png')
        plt.close()
        
        # Plot for cam{i}_rgb.csv
        rgb_df = pd.read_csv(f'../model_output/cam{i}_rgb.csv')
        plt.figure(figsize=(10, 7))
        plt.scatter(rgb_df['Center_X'], rgb_df['Center_Y'], c='b', marker='o')
        plt.title(f'Camera {i} RGB Center_X and Center_Y values')
        plt.xlabel('Center_X')
        plt.ylabel('Center_Y')
        plt.grid(True)
        plt.savefig(f'{output_folder}/cam{i}_rgb_center_x_y.png')
        plt.close()

def calculate_panel_rgb_values(csv_file, image_folder):
    data = pd.read_csv(csv_file)
    results = []

    for index, row in data.iterrows():
        filename = row['Filename']
        center_x = row['Center_X']
        center_y = row['Center_Y']
        
        image_path = os.path.join(image_folder, filename)
        if not os.path.exists(image_path):
            continue
        
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        x_start = max(0, int(center_x) - 5)
        x_end = min(image.shape[1], int(center_x) + 5)
        y_start = max(0, int(center_y) - 5)
        y_end = min(image.shape[0], int(center_y) + 5)
        
        region = image[y_start:y_end, x_start:x_end]
        average_color = region.mean(axis=(0, 1))
        
        results.append({
            'Filename': filename,
            'Center_X': center_x,
            'Center_Y': center_y,
            'Average_R': average_color[2],
            'Average_G': average_color[1],
            'Average_B': average_color[0]
        })
    
    return pd.DataFrame(results)

def process_all_cameras():
    csv_files = [f"../model_output/cam{i}_centers.csv" for i in range(1, 9)]
    image_folders = [f"../model_output/cam{i}" for i in range(1, 9)]

    for csv_file, image_folder in zip(csv_files, image_folders):
        result_df = calculate_panel_rgb_values(csv_file, image_folder)
        print(f"Processed {len(result_df)} images for {csv_file}")
        print(result_df)

if __name__ == "__main__":
    output_folder = '../assets'
    
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Plot center values and save plots
    plot_center_values(output_folder)
    
    # Calculate panel RGB values for all cameras
    process_all_cameras()