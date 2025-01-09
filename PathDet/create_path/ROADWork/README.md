## ROADWork Dataset Curation
Processing ROADWork dataset for generating drivable path trajectory, we have used the following steps:

* STEP 01: Create subdirectories for the following outputs - raw PNG images, trajectory path visualization and trajectory line masks
* STEP 02: Read all JSON files and create a combined JSON data (list of dictionaries)
* STEP 03: Parse JSON data and create drivable path JSON file and Trajecory Images (RGB and Binary)
    * STEP 03(a): Convert JPG to PNG format and store in output directory
    * STEP 03(b): Read Trajectory and process the trajectory points as tuples
    * STEP 03(c): Create Trajectory Overlay and Mask, and save
    * STEP 03(d): Normalize the trajectory points
    * STEP 03(e): Create drivable path JSON file

### Usage:
```bash
usage: process_roadwork.py [-h] --image-dir IMAGE_DIR --annotation-dir ANNOTATION_DIR [--output-dir OUTPUT_DIR]

Process ROADWork dataset - PathDet groundtruth generation

options:
  -h, --help            show this help message and exit
  --image-dir IMAGE_DIR, -i IMAGE_DIR
                        ROADWork Image Datasets directory
  --annotation-dir ANNOTATION_DIR, -a ANNOTATION_DIR
                        ROADWork Trajectory File directory
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        Output directory
```

### Example:
```bash
$ python process_roadwork.py --image-dir ~/datasets/roadwork/traj_images\
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