from venv import create
import os
from subprocess import run

def check_requirements_file() -> bool:
    """Check if the requirements file exists in the current working directory.
    
    Returns:
        bool: True if the requirements file exists, False otherwise.
    """
    return os.path.exists(os.path.join(os.getcwd(), "installation", "requirements.txt"))

def check_virtual_environment() -> bool:
    """Check if the virtual environment exists in the current working directory.
    
    Returns:
        bool: True if the virtual environment exists, False otherwise.
    """
    return os.path.exists("yolo8_venv")

def get_virtual_environment_path() -> str:
    """Get the path to the virtual environment.
    
    Returns:
        str: The path to the virtual environment.
    """
    return os.path.join(os.getcwd(), "yolo8_venv")

def create_virtual_environment() -> bool:
    """Create a virtual environment in the current working directory.
    
    Returns:
        bool: True if the virtual environment is created successfully, False otherwise.
    """
    venv_path: str = get_virtual_environment_path()
    create(venv_path, with_pip=True)
    return check_virtual_environment()

def install_requirements(venv_path: str) -> None:
    """Install the requirements from the requirements file into the virtual environment.
    
    Args:
        venv_path (str): The path to the virtual environment where the requirements will be installed.
    """
    python_executable: str = os.path.join(venv_path, "Scripts", "python.exe")
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

    venv_path: str = get_virtual_environment_path()
    # Install the requirements
    install_requirements(venv_path)
    print("Requirements installed")
    os.sys.exit(0)