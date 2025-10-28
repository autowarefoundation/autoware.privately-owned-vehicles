import json
from PIL import Image, ImageDraw, ImageFont
import os
import argparse

def visualize(dataset_path: str, num_per_category: int):
    """
    Creates a structured set of visualizations, taking a few samples from each
    camera and data split (train/val/test).
    """
    print(f"Creating a structured visualization with {num_per_category} samples per category.")
    
    base_output_dir = os.path.join(dataset_path, 'test_view')
    annotation_dir = os.path.join(dataset_path, 'annotations')

    if not os.path.isdir(annotation_dir):
        print(f"Error: Annotation directory not found at {annotation_dir}")
        print("Please run the annotation generation script first.")
        return

    # Dynamically find all camera folders
    camera_folders = [d for d in os.listdir(annotation_dir) if os.path.isdir(os.path.join(annotation_dir, d))]
    splits = ['train', 'val', 'test']

    print(f"Found camera folders: {camera_folders}")

    # Loop through every camera and every split
    for camera in camera_folders:
        for split in splits:
            split_dir = os.path.join(annotation_dir, camera, split)
            if not os.path.isdir(split_dir):
                continue

            # Find all annotation files in this specific category
            ann_files = [os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith('.json')]
            if not ann_files:
                continue
                
            print(f"\n--- Processing {split.upper()}/{camera} ---")

            # Take the first few files for visualization
            for i, ann_path in enumerate(ann_files[:num_per_category]):
                
                # Construct the corresponding image path by replacing 'annotations' and the split
                # e.g. .../annotations/CAM_FRONT/train/file.json -> .../samples/CAM_FRONT/file.jpg
                img_path_str = ann_path.replace(f'/{split}/', '/')
                img_path_str = img_path_str.replace('/annotations/', '/samples/', 1)
                img_path_str = os.path.splitext(img_path_str)[0] + '.jpg'
                
                if not os.path.exists(img_path_str):
                    print(f"Warning: Could not find corresponding image for {ann_path}. Skipping.")
                    continue
                
                # Load data
                with open(ann_path, 'r') as f:
                    annotations = json.load(f)
                image = Image.open(img_path_str)
                draw = ImageDraw.Draw(image)

                try:
                    font = ImageFont.truetype("dejavusans.ttf", size=20)
                except IOError:
                    font = ImageFont.load_default()
                
                # Draw annotations
                for ann in annotations:
                    bbox = ann['bbox']
                    category = ann['category']
                    draw.rectangle(bbox, outline="lime", width=3)
                    draw.text((bbox[0] + 5, bbox[1] + 5), category, fill="lime", font=font)
                
                # Create the structured output path
                output_dir = os.path.join(base_output_dir, camera, split)
                os.makedirs(output_dir, exist_ok=True)
                output_filename = f'visualization_{i}.jpg'
                output_path = os.path.join(output_dir, output_filename)
                
                image.save(output_path)
                print(f"-> Saved to {output_path}")

    print(f"\nProcessing complete. All visualized images are in the '{base_output_dir}' directory.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize a structured sample of object annotations for a nuImages dataset.")
    parser.add_argument('dataset_path', type=str, help='Path to the dataset directory (e.g., nuimages-v1.0-all-samples).')
    parser.add_argument('--num', type=int, default=3, help='Number of images to visualize per category (camera/split).')
    args = parser.parse_args()
    
    visualize(args.dataset_path, args.num)
