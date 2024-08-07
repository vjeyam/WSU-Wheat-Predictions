import os

def rename_files(folder_path):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)
    
    # Sort files to ensure consistent numbering
    files.sort()
    
    # Initialize counter
    i = 1 # Change this to the next number in the sequence
    
    # Iterate through the files and rename them
    for filename in files:
        # Check if the filename starts with 'date_'
        if filename.startswith('date_'):
            # Construct the new file name
            new_filename = f'cam1 ({i}).png' # Change this to cam2, cam3, cam4, etc. as needed
            
            # Get the full file path
            old_file = os.path.join(folder_path, filename)
            new_file = os.path.join(folder_path, new_filename)
            
            # Rename the file
            os.rename(old_file, new_file)
            
            # Increment the counter
            i += 1
        else:
            continue

# Example usage
folder_path = 'path/to/folder' # Change this to the path of the folder containing the files
rename_files(folder_path)