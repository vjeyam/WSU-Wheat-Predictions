# WSU Summer REEU 2024: Wheat Predictions

The objective of this project is to extract crop features, such as the Vegetative Index (VI), from multispectral images of crops to predict wheat health and yield in real-time. This tool will aid breeding programs by reducing costs and resource usage.

If you would like to see the in depth process, [click here](Solution.md).

## Download the Project

You may either download the project as a ZIP or lcone the repository to your local machine:

```bash
$ git clone https://github.com/vjeyam/WSU-Wheat-Predictions.git
```

## Installation and Setup: Conda

### Step 1: Install Conda

If you haven't installed Conda yet, you can download and install it from the [Anaconda website](https://www.anaconda.com/products/distribution) or [Miniconda website](https://docs.conda.io/en/latest/miniconda.html), depending on your preference and system requirements.

### Step 2: Create a Conda Environment

Create a new Conda environment for the project using the following command. Please note that any version of python greater than 3.9 should work.

```bash
$ conda create --name wheat python=3.9
```

Replace `wheat` with the desired name for your environment.

### Step 3: Activate the Environment

Activate the newly created environment with the following command:

```bash
$ conda activate wheat
```

### Step 4: Install Dependencies

Navigate outside the `src` directory and install the required dependencies using `pip` or `conda`:

```bash
$ cd ..
$ pip install -r requirements.txt
```

### Step 5: Verify Installation

Verifity that the environmnet and dependencies are install correclty:

```bash
$ python --version
```

This should display the Python version installed in your Conda environment.

```bash
$ conda list
```

This command will list all packages installed in the current environment.

### Step 6: Run the Project
To run the project, navigate to the `src` directory and execute the main script:

```bash
$ cd src
$ python image_segmentation.py
```

This will start the application, and you can follow the on-screen instructions to use the tool.

## Usage

### Step 1: Load the Image

Load the multispectral image of the wheat crop that you want to analyze. The tool supports various image formats, including JPEG, PNG, and TIFF.

### Step 2: Select the Region of Interest (ROI)

Select the region of interest (ROI) in the image. You can use the provided tools to draw a bounding box around the area you want to analyze.

### Step 3: Extract Features

The tool will automatically extract the relevant features from the selected ROI, including the Vegetative Index (VI) and other crop health indicators.

### Step 4: Analyze Results

The extracted features will be displayed in a user-friendly interface, allowing you to analyze the crop health and yield predictions. You can also save the results for further analysis or reporting.

### Step 5: Save Results

You can save the extracted features and analysis results to a file for future reference. The tool supports various file formats, including CSV and Excel.
