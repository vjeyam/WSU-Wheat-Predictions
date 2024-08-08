from venv import create
import os
from subprocess import run

def check_requirements_file():
    """Check if the requirements file exists in the current working directory."""
    return os.path.exists(os.path.join(os.getcwd(), "installation", "../../requirements.txt"))

def check_virtual_environment():
    """Check if the virtual environment exists in the current working directory."""
    return os.path.exists("yolo8_venv")

def get_virtual_environment_path():
    """Get the path to the virtual environment."""
    return os.path.join(os.getcwd(), "yolo8_venv")

def create_virtual_environment():
    """Create a virtual environment in the current working directory."""
    venv_path = get_virtual_environment_path()
    create(venv_path, with_pip=True)
    return check_virtual_environment()

def install_requirements(venv_path):
    """Install the requirements file in the virtual environment."""
    python_executable = os.path.join(venv_path, "Scripts", "python.exe")
    run([python_executable, "-m", "pip", "install", "-r", os.path.join("installation", "requirements.txt")])

if __name__ == "__main__":
    # Check if the requirements file exists
    if not check_requirements_file():
        print("Requirements file not found")
        os.sys.exit(1)
    print("Requirements file found")

    # Check if the virtual environment exists
    if check_virtual_environment():
        print("Virtual environment already exists, installing requirements")
    else:
        print("Virtual environment not found, creating virtual environment")
        if create_virtual_environment():
            print("Virtual environment created")
        else:
            print("Virtual environment not created")
            os.sys.exit(1)

    venv_path = get_virtual_environment_path()
    # Install the requirements
    install_requirements(venv_path)
    print("Requirements installed")
    os.sys.exit(0)