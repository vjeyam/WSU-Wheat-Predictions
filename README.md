# WSU Summer REEU 2024: Wheat Predictions

The objective of this project is to extract crop features, such as the Vegetative Index (VI), from multispectral images of crops to predict wheat health and yield in real-time. This tool will aid breeding programs by reducing costs and resource usage.

If you would like to see the in depth process, [click here](Solution.md).

## Download the Project

Clone the repository to your local machine:

```bash
git clone https://github.com/vjeyam/WSU-Wheat-Predictions.git
cd WSU-Wheat-Predictions
```

## Environment Setup

### Option 1: Using Conda (Recommended)

Create a new Conda environment for the project using the following command. Please note that any version of python greater than 3.9 should work.

```bash
conda create --name wheat python=3.9 -y
conda activate wheat
```

Install Dependencies

```bash
# From the root directory of the project
pip install -r installation/requirements.txt
```

### Option 2: Using Virtualenv

Place setup_env.py at the repo root (already included). It expects installation/requirements.txt and creates yolo8_venv/ alongside it

```bash
# From repo root
python3 setup_env.py

# Activate the venv

# Windows:
.\yolo8_venv\Scripts\activate

# macOS/Linux/WSL:
source yolo8_venv/bin/activate
```

Notes:

* The script checks for installation/requirements.txt and installs from there.
* If you see errors about subprocess32, remove it (Python 3 doesn’t need it). It’s currently listed in your requirements.

### PyTorch Installation:

Pick one:

```bash
# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.4 (example)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Then install the rest (as above).

## What’s included

* Image processing & indices: src/image_segmentation.py (splits NIR/RGB, computes SCI/GNDVI, saves plots).
* Panel detection model (YOLOv8): training/validation/testing scripts.
<!-- * (Wheat model scripts are similar; see “Datasets: wheat” when you add those files.) -->

## Usage

From the project root, run:

```bash
cd src
python image_segmentation.py
```

That launches the processing flow and saves outputs to `assets/` (center plots, SCI/GNDVI visualizations). This matches your original step

> Behind the scenes the script reads model_output/cam{i}_nir.csv and cam{i}_rgb.csv, clamps the sampling windows, and normalizes the SCI/GNDVI images to avoid NaN/Inf warnings. (These robust guards are implemented in the current script.)

## Troubleshooting

* Ultralytics settings banner: harmless info on first run; you can ignore it.
* “Source not found” in test.py: pass --source datasets/YOLOv8_TRP/test/images, or use the robust resolver in the updated script (it tries multiple candidate paths when path: is relative).
* NaN/Inf warnings when saving SCI/GNDVI: the current script bounds windows and normalizes before Image.fromarray, preventing these warnings.

## Credits

WSU Summer REEU 2024 program, with special thanks to Dr. Sindhuja Sankaran.
