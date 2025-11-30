import os
import json

def reorder_files_and_json(base_dir, subdirs, json_mapping_path, json_metadata_path):
    # Step 1: Find all files and synchronize deletions across subdirectories
    # Step 1: Collect filenames from all subdirectories
    all_subdirs = [os.path.join(base_dir, subdir) for subdir in subdirs]
    common_files = None

    for subdir in all_subdirs:
        subdir_files = set(os.listdir(subdir))
        if common_files is None:
            common_files = subdir_files
        else:
            common_files &= subdir_files  # Intersection to keep only common files
    
    # Step 2: Remove files not in the common set
    for subdir in all_subdirs:
        current_files = set(os.listdir(subdir))
        for file in current_files - common_files:
            file_path = os.path.join(subdir, file)
            os.remove(file_path)
    
    # Step 2: Get remaining filenames and rename them to maintain order
    filenames = sorted(os.listdir(os.path.join(base_dir, subdirs[0])))
    filename_mapping = {}
    for new_index, filename in enumerate(filenames):
        new_name = f"{new_index:05}.png"  # Change extension if needed
        for subdir in all_subdirs:
            old_path = os.path.join(subdir, filename)
            new_path = os.path.join(subdir, new_name)
            os.rename(old_path, new_path)
        filename_mapping[filename.replace('.png','')] = new_name.replace('.png','')  # Map old filename to new filename
    
    # Step 3: Update the JSON mapping file
    if os.path.exists(json_mapping_path):
        with open(json_mapping_path, 'r') as f:
            mapping_data = json.load(f)
        
        new_mapping_data = {}
        for old_name, actual_filename in mapping_data.items():
            if old_name in filename_mapping:
                new_name = filename_mapping[old_name]
                new_mapping_data[new_name] = actual_filename

        with open(json_mapping_path, 'w') as f:
            json.dump(new_mapping_data, f, indent=4)
    else:
        print(f"JSON mapping file not found: {json_mapping_path}")

    # Step 4: Update the JSON metadata file
    if os.path.exists(json_metadata_path):
        with open(json_metadata_path, 'r') as f:
            metadata_data = json.load(f)
        
        new_metadata_data = {"files": metadata_data["files"], "data": {}}
        for new_index, old_key in enumerate(sorted(metadata_data["data"].keys())):
            if old_key in filename_mapping:  # Check if the key corresponds to a valid file
                new_key = filename_mapping[old_key]
                new_metadata_data["data"][new_key] = metadata_data["data"][old_key]

        with open(json_metadata_path, 'w') as f:
            json.dump(new_metadata_data, f, indent=4)
    else:
        print(f"JSON metadata file not found: {json_metadata_path}")

if __name__ == '__main__':
    base_directory = "path/to/processed/data"
    subdirectories = ["segmentation", "image","visualization"] 
    json_mapping_path = os.path.join(base_directory,'name.json')  # JSON containing file-name mapping
    json_metadata_path = os.path.join(base_directory,'drivable_path.json')  # JSON containing metadata

    reorder_files_and_json(base_directory, subdirectories, json_mapping_path, json_metadata_path)
