# WSU Summer REEU 2024: Wheat Predictions

The objective of this project is to extract crop features, such as the Vegetative Index (VI), from multispectral images of crops to predict wheat health and yield in real-time. This tool will aid breeding programs by reducing costs and resource usage.

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