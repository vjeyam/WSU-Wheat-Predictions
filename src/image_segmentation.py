import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from typing import Tuple, Dict

def process_rgb_images(input_folder, output_folder_rgb) -> None:
    cam_folders = [f'cam{i}' for i in range(1, 9)]
    for cam in cam_folders:
        cam_path = os.path.join(input_folder, cam)
        output_cam_rgb_path = os.path.join(output_folder_rgb, f'{cam}_rgb')
        os.makedirs(output_cam_rgb_path, exist_ok=True)
        for filename in os.listdir(cam_path):
            if filename.endswith('.png'):
                image_path = os.path.join(cam_path, filename)
                img = Image.open(image_path)
                if img.size != (1280, 928):
                    print(f"Skipping {image_path}: unexpected image size {img.size}")
                    continue
                right_half = img.crop((640, 0, 1280, 928))
                right_half.save(os.path.join(output_cam_rgb_path, filename))
                print(f"Processed right side of {filename}")

def process_nir_images(input_folder, output_folder_ir) -> None:
    cam_folders = [f'cam{i}' for i in range(1, 9)]
    for cam in cam_folders:
        cam_path = os.path.join(input_folder, cam)
        output_cam_ir_path = os.path.join(output_folder_ir, f'{cam}_nir')
        os.makedirs(output_cam_ir_path, exist_ok=True)
        for filename in os.listdir(cam_path):
            if filename.endswith('.png'):
                image_path = os.path.join(cam_path, filename)
                img = Image.open(image_path)
                if img.size != (1280, 928):
                    print(f"Skipping {image_path}: unexpected image size {img.size}")
                    continue
                left_half = img.crop((0, 0, 640, 928))
                left_half.save(os.path.join(output_cam_ir_path, filename))
                print(f"Processed left side of {filename}")

def reorder_csv_files(model_output_dir: str) -> None:
    prefixes = [f'cam{i}' for i in range(1, 9)]
    suffixes = ['_nir.csv', '_rgb.csv']
    all_files = os.listdir(model_output_dir)
    csv_files = [f for f in all_files if any(f.startswith(prefix) and f.endswith(suffix)
                                             for prefix in prefixes for suffix in suffixes)]
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(model_output_dir, csv_file))
        df = df.dropna(subset=['Center_X', 'Center_Y']).sort_values(by='Filename')
        df.to_csv(os.path.join(model_output_dir, csv_file), index=False)
        print(f"Processed {csv_file}")

def plot_center_coordinates(model_output_dir: str, assets_dir: str) -> None:
    for i in range(1, 9):
        for mode in ['nir', 'rgb']:
            df = pd.read_csv(f'{model_output_dir}/cam{i}_{mode}.csv')
            plt.figure(figsize=(10, 7))
            plt.scatter(df['Center_X'], df['Center_Y'], c='r' if mode == 'nir' else 'b', marker='o')
            plt.title(f'Camera {i} {mode.upper()} Center_X and Center_Y values')
            plt.xlabel('Center_X')
            plt.ylabel('Center_Y')
            plt.grid(True)
            os.makedirs(assets_dir, exist_ok=True)
            plt.savefig(f'{assets_dir}/cam{i}_{mode}_center_x_y.png')
            plt.close()

def calculate_panel_rgb_values(csv_file: str, image_folder: str) -> pd.DataFrame:
    data = pd.read_csv(csv_file)
    results = []
    for _, row in data.iterrows():
        image_path = os.path.join(image_folder, row['Filename'])
        if not os.path.exists(image_path):
            continue
        image = cv2.imread(image_path)
        if image is None:
            continue
        x_start, x_end = max(0, int(row['Center_X']) - 5), min(image.shape[1], int(row['Center_X']) + 5)
        y_start, y_end = max(0, int(row['Center_Y']) - 5), min(image.shape[0], int(row['Center_Y']) + 5)
        region = image[y_start:y_end, x_start:x_end]
        if region.size == 0:
            continue
        avg_color = region.mean(axis=(0, 1))
        results.append({
            'Filename': row['Filename'],
            'Center_X': row['Center_X'],
            'Center_Y': row['Center_Y'],
            'Average_R': avg_color[2],
            'Average_G': avg_color[1],
            'Average_B': avg_color[0]
        })
    return pd.DataFrame(results)

calibration_data: Dict[str, Dict[str, float]] = {
    f'cam{i}': {'Blue': 10+i, 'Red': 12+i, 'Green': 12+i} for i in range(1, 9)
}

def _to_vis(a: np.ndarray) -> np.ndarray:
    a = np.array(a, dtype=np.float64)
    a = np.where(np.isfinite(a), a, np.nan)
    mn = np.nanmin(a)
    mx = np.nanmax(a)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < 1e-12:
        return np.zeros_like(a, dtype=np.uint8)
    a = (a - mn) / (mx - mn) * 255.0
    a = np.nan_to_num(a, nan=0.0, posinf=255.0, neginf=0.0).astype(np.uint8)
    return a

def vi(img: np.ndarray, nir_x: float, nir_y: float, rgb_x: float, rgb_y: float, cam_name: str) -> Tuple[np.ndarray, np.ndarray]:
    rgb_x, rgb_y, nir_x, nir_y = map(lambda v: int(round(v)), (rgb_x, rgb_y, nir_x, nir_y))
    nir = img[:, :640, :].astype(np.float64)
    rgb = img[:, 640:1280, :].astype(np.float64)
    def bounded_region(a, cx, cy, half=5):
        h, w = a.shape[:2]
        xs, xe = max(0, cx-half), min(w, cx+half)
        ys, ye = max(0, cy-half), min(h, cy+half)
        return a[ys:ye, xs:xe]
    rgb_region = bounded_region(img, rgb_x, rgb_y, half=5)
    nir_region = bounded_region(img, nir_x, nir_y, half=5)
    rgb_avg = np.mean(rgb_region, axis=(0, 1)) if rgb_region.size else np.array([1e-6,1e-6,1e-6])
    nir_avg = np.mean(nir_region, axis=(0, 1)) if nir_region.size else np.array([1e-6,1e-6,1e-6])
    red, green = rgb[:, :, 2], rgb[:, :, 1]
    with np.errstate(divide='ignore', invalid='ignore'):
        num = (red / rgb_avg[2]) - (green / rgb_avg[1])
        den = (red / rgb_avg[2]) + (green / rgb_avg[1])
        den = np.where(np.abs(den) < 1e-12, np.nan, den)
        sci = num / den
    nir_red, nir_green = nir[:, :, 2], nir[:, :, 1]
    calib = calibration_data.get(cam_name, {'Red':100,'Green':100})
    with np.errstate(divide='ignore', invalid='ignore'):
        nir_red = nir_red / (nir_avg[2] * (calib['Red'] / 100.0))
        nir_green = nir_green / (nir_avg[1] * (calib['Green'] / 100.0))
        d = nir_red + nir_green
        d = np.where(np.abs(d) < 1e-12, np.nan, d)
        gndvi = (nir_red - nir_green) / d
    return sci, gndvi

def process_cameras(data_dir: str, model_output_dir: str, assets_dir: str) -> None:
    for i in range(1, 9):
        cam_name = f'cam{i}'
        rgb_csv = f'{model_output_dir}/{cam_name}_rgb.csv'
        nir_csv = f'{model_output_dir}/{cam_name}_nir.csv'
        if not (os.path.exists(rgb_csv) and os.path.exists(nir_csv)):
            continue
        df_rgb = pd.read_csv(rgb_csv)
        df_nir = pd.read_csv(nir_csv)
        if df_rgb.empty or df_nir.empty:
            continue
        rgb_x, rgb_y = df_rgb.iloc[0][['Center_X','Center_Y']].astype(float).tolist()
        nir_x, nir_y = df_nir.iloc[0][['Center_X','Center_Y']].astype(float).tolist()
        cam_folder = os.path.join(data_dir, cam_name)
        for img_file in os.listdir(cam_folder):
            if img_file.lower().endswith('.png'):
                img = np.array(Image.open(os.path.join(cam_folder, img_file))).astype(np.float64)
                sci, gndvi = vi(img, nir_x, nir_y, rgb_x, rgb_y, cam_name)
                base_name = os.path.splitext(img_file)[0]
                os.makedirs(f'{assets_dir}/vi/sci/{cam_name}', exist_ok=True)
                os.makedirs(f'{assets_dir}/vi/gndvi/{cam_name}', exist_ok=True)
                Image.fromarray(_to_vis(sci)).save(f'{assets_dir}/vi/sci/{cam_name}/{base_name}_sci.png')
                Image.fromarray(_to_vis(gndvi)).save(f'{assets_dir}/vi/gndvi/{cam_name}/{base_name}_gndvi.png')

def main():
    input_folder = '../data'
    model_output_dir = '../model_output'
    assets_dir = '../assets'
    process_nir_images(input_folder, input_folder)
    process_rgb_images(input_folder, input_folder)
    reorder_csv_files(model_output_dir)
    plot_center_coordinates(model_output_dir, assets_dir)
    for i in range(1, 9):
        csv_path = f"{model_output_dir}/cam{i}_rgb.csv"
        image_folder = f"{input_folder}/cam{i}"
        df = calculate_panel_rgb_values(csv_path, image_folder)
        if not df.empty:
            df.to_csv(f"{model_output_dir}/cam{i}_panel_rgb_values.csv", index=False)
    process_cameras(input_folder, model_output_dir, assets_dir)

if __name__ == "__main__":
    main()