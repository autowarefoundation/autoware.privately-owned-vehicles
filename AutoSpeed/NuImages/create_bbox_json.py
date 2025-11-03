import json
import os
import argparse
from nuimages import NuImages

def process_dataset(dataset_path: str, metadata_path: str, versions: list):
    """
    Generates single-class 'object' annotation files for given nuImages dataset splits,
    organizing them into train/val/test subdirectories.
    """
    print(f"Processing dataset at: {dataset_path}")
    
    for version in versions:
        split_name = version.split('-')[-1] # Extracts 'train', 'val', or 'test'
        print(f"\n--- Processing version: {version} (split: {split_name}) ---")

        try:
            nui = NuImages(dataroot=metadata_path, version=version, verbose=False, lazy=False)
        except AssertionError:
            print(f"Warning: Could not load metadata version '{version}'. Skipping.")
            continue

        processed_count = 0
        for sample in nui.sample:
            key_camera_token = sample['key_camera_token']
            sample_data = nui.get('sample_data', key_camera_token)
            
            relative_image_path = sample_data['filename']
            full_image_path = os.path.join(dataset_path, relative_image_path)

            if os.path.exists(full_image_path):
                # Construct the new, structured output path
                # e.g., 'samples/CAM_FRONT/file.jpg' -> 'annotations/CAM_FRONT/train/file.json'
                
                # First, replace 'samples' with 'annotations'
                relative_ann_path = relative_image_path.replace('samples/', 'annotations/', 1)
                
                # Insert the split name into the path
                dir_name, file_name = os.path.split(relative_ann_path)
                structured_dir = os.path.join(dir_name, split_name)
                
                # Recombine and change extension
                base_name, _ = os.path.splitext(file_name)
                output_json_path = os.path.join(dataset_path, structured_dir, base_name + '.json')
                
                os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

                # Extract and write annotation data
                object_tokens, _ = nui.list_anns(sample['token'])
                image_annotations = []
                for obj_token in object_tokens:
                    obj_ann = nui.get('object_ann', obj_token)
                    image_annotations.append({
                        "category": "object",
                        "bbox": obj_ann['bbox']
                    })
                
                with open(output_json_path, 'w') as f:
                    json.dump(image_annotations, f, indent=4)
                
                processed_count += 1

        print(f"Generated {processed_count} annotation files for the '{split_name}' split.")

    print(f"\nProcessing complete for all versions.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate structured (train/val/test) single-class object annotations for nuImages.")
    parser.add_argument('dataset_path', type=str, help="Path to the directory containing the 'samples' folder (e.g., nuimages-v1.0-all-samples).")
    parser.add_argument('metadata_path', type=str, help="Path to the directory containing the nuImages metadata (e.g., nuimages-v1.0-all-metadata).")
    
    VERSIONS_TO_PROCESS = ['v1.0-train', 'v1.0-val', 'v1.0-test']

    args = parser.parse_args()
    
    process_dataset(args.dataset_path, args.metadata_path, VERSIONS_TO_PROCESS)
