import os
import shutil
from math import ceil

def divide_files_into_subdirs(directory: str, subdir_count: int = 10):
    """
    Divide files in the specified directory into a specified number of subdirectories.
    
    :param directory: Path to the directory containing files.
    :param subdir_count: Number of subdirectories to create (default is 5).
    """
    # List all files in the directory (exclude subdirectories)
    all_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # Calculate the number of files per subdirectory
    num_files_per_subdir = ceil(len(all_files) / subdir_count)
    
    # Create subdirectories
    subdirs = []
    for i in range(subdir_count):
        subdir_path = os.path.join(directory, f"{i+1}")
        os.makedirs(subdir_path, exist_ok=True)
        subdirs.append(subdir_path)
    
    # Distribute files across subdirectories
    for idx, file in enumerate(all_files):
        subdir_index = idx // num_files_per_subdir  # Determine the subdirectory index
        if subdir_index < subdir_count:
            src_path = os.path.join(directory, file)
            dst_path = os.path.join(subdirs[subdir_index], file)
            shutil.move(src_path, dst_path)
    
    print(f"Files successfully divided into {subdir_count} subdirectories.")

# Usage
directory = "path/to/audited/data"
divide_files_into_subdirs(directory)
