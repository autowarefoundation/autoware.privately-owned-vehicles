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

### Dataset Curateion Workflow
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
usage: process_roadwork.py [-h] --image-dir IMAGE_DIR --annotation-dir ANNOTATION_DIR [--output-dir OUTPUT_DIR] [--display DISPLAY]

Process ROADWork dataset - PathDet groundtruth generation

options:
  -h, --help            show this help message and exit
  --image-dir IMAGE_DIR, -i IMAGE_DIR
                        ROADWork Image Datasets directory. DO NOT include subdirectories or files.
  --annotation-dir ANNOTATION_DIR, -a ANNOTATION_DIR
                        ROADWork Trajectory Annotations Parent directory. Do not include subdirectories or files.
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