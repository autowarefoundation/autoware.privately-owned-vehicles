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
    1. `raw PNG images`
    2. `trajectory path visualization`
    3. `trajectory line masks`
* **STEP 02:** Read all `JSON` files and create a combined `JSON` data (list of dictionaries)
* **STEP 03:** Parse `JSON` data and create drivable path `JSON` file and Trajecory `Images` (RGB and Binary)
    * STEP 03(a): Read `Trajectory` data and process the points as tuples and integer
    * STEP 03(b): Create `Trajectory Overlay`
    * STEP 03(c): Crop the image to aspect ratio `2:1` and convert from `JPG` to `PNG` format and store in output directory
    * STEP 03(d): Create Cropped Image Mask using `STEP 03(b) - 03(c)`    
    * STEP 03(e): `Normalize` the trajectory points
    * STEP 03(f): Build `Data Structure` for final `JSON` file
* STEP 04: Create drivable path `JSON` file


### Usage:
```bash
usage: process_roadwork.py [-h] --image-dir IMAGE_DIR --annotation-dir ANNOTATION_DIR [--output-dir OUTPUT_DIR]

Process ROADWork dataset - PathDet groundtruth generation

options:
  -h                                    Show this help message and exit
  --help

  --image-dir IMAGE_DIR                 ROADWork Image Parent directory
  -i IMAGE_DIR

  --annotation-dir ANNOTATION_DIR       ROADWork Trajectory Annotations Parent directory. 
  -a ANNOTATION_DIR                     Do not include subdirectories or files.
  
  --output-dir OUTPUT_DIR               Output directory 
  -o OUTPUT_DIR

```

### Example:
```bash
$ python process_roadwork.py\
> --image-dir ~/datasets/roadwork/traj_images\
> --annotation-dir ~/datasets/roadwork/traj_annotations\
> --output-dir ~/tmp/output
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