## ROADWork Dataset Curation

### Dataset Overview
* Number of Trajectory Images: 5430 (with Temporal Downsampling of 10)
* Number of Cities: 18
* Image Format: .jpg
* Image Frame Rates: 5 FPS
* Image Capture: iPhone 14 Pro Max paired with a Bluetooth remote trigger
* Images captured from two sources: 
    * Robotics Institute, Carnegie Mellon University
    * Michelin Mobility Intelligence (MMI) (formerly RoadBotics) Open Dataset.
* Dataset link: [ROADWork Dataset](https://kilthub.cmu.edu/articles/dataset/ROADWork_Data/26093197)

### Dataset Curation Workflow
Processing ROADWork dataset for generating drivable path trajectory, we have used the following steps:

* **STEP 01:** Create subdirectories for the following outputs:
    1. `Original Images in PNG format`
    2. `Trajectory path Overlay in PNG format`
    3. `Trajectory path Binary Mask`

* **STEP 02:** Read all `JSON` files and create a combined `JSON` data (list of dictionaries)
* **STEP 03:** Parse `JSON` data and create drivable path `JSON` file and Trajectory `Images` (RGB and Binary)
    * **STEP 03(a):** Process the `Trajectory Points` as tuples
    * **STEP 03(b):** Crop the original image to aspect ratio `2:1` and convert from `JPG` to `PNG` format
    * **STEP 03(c):** Normalize the `Trajectory Points` and filter out the points outside the range [0, 1]
    * **STEP 03(d):** Create `Trajectory Overlay` and crop it to aspect ratio `2:1`
    * **STEP 03(e):** Create `Cropped Trajectory Binary Mask` with aspect ratio `2:1`
    * **STEP 03(f):** Save all images (original, cropped, and overlay) in the output directory
    * **STEP 03(g):** Build `Data Structure` for final `JSON` file
* **STEP 04:** Create drivable path `JSON` file


### Usage:
```bash
usage: process_roadwork.py [-h] --image-dir IMAGE_DIR --annotation-dir ANNOTATION_DIR [--crop-size CROP_SIZE] [--output-dir OUTPUT_DIR]

Process ROADWork dataset - Egopapth groundtruth generation

options:
  -h, --help            show this help message and exit
  --image-dir IMAGE_DIR, -i IMAGE_DIR
                        ROADWork Image Datasets directory. DO NOT include subdirectories or files.
  --annotation-dir ANNOTATION_DIR, -a ANNOTATION_DIR
                        ROADWork Trajectory Annotations Parent directory. Do not include subdirectories or files.
  --crop-size CROP_SIZE, -c CROP_SIZE
                        Minimum pixels to crop from the bottom
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        Output directory for image, segmentation, and visualization 

```

### Example:
```bash
$ python process_roadwork.py\
> --image-dir ~/autoware_datasets/roadwork/traj_images/\
> --annotation-dir ~/autoware_datasets/roadwork/traj_annotations/\
> --output-dir ~/tmp/output --display rgb
```

### Dataset Audit Workflow
In ROADWork dataset, we need to filter out images which are irrelevant to our objective. For filtering out the images, we have set the following criteria:

* Intersection
* Road blocks or Dead-ends
* Lane changing
* Right turn in the intersection
* Left turn in the intersection

Therefore, for auditing the dataset, we used `audit_roadwork.py` script using the following steps:

* **STEP 00:** Remove irrelevant images from `visualization` directory by checking the trajectory path in the image
* **STEP 01:** List all audited cropped overlay images
* **STEP 02:** Create subdirectories for the following outputs:
    1. `Original Images in PNG format`
    2. `Trajectory path Overlay in PNG format`
    3. `Trajectory path Binary Mask`
* **STEP 03:** Read and parse the JSON file `drivable_path.json`
    * **STEP 03(a):** Iterate each old `JSONID` and generate a new one (as some images are removed and JSONIDs are now inconsistent) 
    * **STEP 03(b):** copy the images (raw, overlay and binary mask) to the output directory
    * **STEP 03(c):** Get the meta data (trajectory points, width and height of image) of each JSONID and store in a list
    * **STEP 03(c):** Build `Data Structure` for final `JSON` file
* **STEP 04:** Create an updated `drivable_path.json` file
* **OPTIONAL STEP:** Create JSON file for mapping image ID to JSON ID

### Usage:
```bash
usage: audit_roadwork.py [-h] --dataset-dir DATASET_DIR [--output-dir OUTPUT_DIR]

Process ROADWork dataset - EgoPath groundtruth generation

options:
  -h, --help            show this help message and exit
  --dataset-dir DATASET_DIR, -d DATASET_DIR
                        ROADWork Audited Cropped Image Dataset directory. DO NOT include subdirectories or files.
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        Output directory for JSON File
```

### Example:
```bash
$ python audit_roadwork.py\
> --dataset-dir ~/autoware_datasets/roadwork/\
> --output-dir ~/tmp/output

### ROADWork Dataset Outputs

* RGB image in PNG Format
* Drivable path trajectories in JSON Format
* Binary Drivable Path Mask in PNG format
* Drivable Path Mask draw on top of RGB image in PNG format (not used during training, only for data auditing purposes)


### Generate output structure
```
    --output_dir
        |----image
        |----segmentation
        |----visualization
        |----drivable_path.json
```