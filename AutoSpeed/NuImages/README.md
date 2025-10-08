# nuImages Dataset Processing

This directory contains scripts for parsing the nuImages dataset, generating annotations, and visualizing the results.

## Folder Structure

For the scripts to work correctly, your nuImages dataset should be organized as follows:

```
/path/to/your/data/
│
├── nuimages-v1.0-all-metadata/      (nuImages metadata)
│   ├── v1.0-test/
│   ├── v1.0-train/
│   ├── v1.0-val/
│   └── ... (other metadata files)
│
└── nuimages-v1.0-all-samples/       (nuImages samples/images)
    ├── samples/
    │   ├── CAM_FRONT/
    │   ├── CAM_BACK/
    │   └── ... (other camera folders)
    │
    ├── annotations/  (This will be created by create_bbox_json.py)
    │   ├── CAM_FRONT/
    │   │   ├── train/
    │   │   ├── val/
    │   │   └── test/
    │   └── ... 
    │
    └── test_view/      (This will be created by test_Visualization.py)
        ├── CAM_FRONT/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── ...
```

-   **`nuimages-v1.0-all-metadata/`**: Contains the official nuImages metadata and JSON files.
-   **`nuimages-v1.0-all-samples/`**: Contains the `samples` directory with all the images. The `annotations` and `test_view` directories will be created by the scripts.

## Scripts

### 1. `create_bbox_json.py`

This script parses the nuImages dataset and creates JSON annotation files for each image. The annotations are for a single class, "object", and include bounding box information.

**How it Works:**

The script iterates through the specified versions of the dataset (e.g., `v1.0-train`, `v1.0-val`, `v1.0-test`). For each image sample, it retrieves the associated object annotations and saves them as a JSON file. The output JSON files are organized into a new `annotations` directory, with subdirectories for each camera and data split (train/val/test).

**Usage:**

```bash
python3 create_bbox_json.py /path/to/nuimages-v1.0-all-samples /path/to/nuimages-v1.0-all-metadata
```

-   `dataset_path`: Path to the directory containing the `samples` folder.
-   `metadata_path`: Path to the directory containing the nuImages metadata.

### 2. `test_Visualization.py`

This script allows you to visualize the generated annotations on the images. It creates a `test_view` directory with sample images that have bounding boxes drawn on them.

**How it Works:**

The script scans the `annotations` directory to find camera and data split subdirectories. For each category (e.g., `CAM_FRONT/train`), it takes a specified number of samples, draws the bounding boxes from the corresponding JSON file onto the image, and saves the result in the `test_view` directory.

**Usage:**

```bash
python3 test_Visualization.py /path/to/nuimages-v1.0-all-samples --num 5
```

-   `dataset_path`: Path to the dataset directory (the same one used for the creation script).
-   `--num` (optional): The number of images to visualize for each category (camera/split). The default is 3.

This will create visualizations in `nuimages-v1.0-all-samples/test_view`.
