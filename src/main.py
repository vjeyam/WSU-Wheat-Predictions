import subprocess
import sys
import os

"""Ensure that the virtual environment is set up and dependencies are installed."""
def setup_environment():
    setup_script_path = os.path.join(os.getcwd(), "src", "scripts", "setup_env.py")
    
    # Run the setup_env.py script
    result = subprocess.run([sys.executable, setup_script_path], capture_output=True, text=True)
    
    # Check the result
    if result.returncode != 0:
        print("Error setting up the environment:\n", result.stderr)
        sys.exit(1)
    
    print("Environment setup successfully")

def main():
    setup_environment()

    # Run the main script
    print("Running main script...")

if __name__ == "__main__":
    main()