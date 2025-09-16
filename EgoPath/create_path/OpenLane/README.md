# OpenLane Dataset Processing Script


## Overview

OpenLane is a large-scale benchmark for lane detection and topology estimation, widely used in autonomous driving and ADAS research. The dataset features diverse road scenarios, complex lane topologies, and high-resolution images. This script suite provides tools for parsing, preprocessing, and transforming OpenLane data for downstream tasks such as drivable path detection, BEV (Bird-Eyes-View) transformation, and model training.


## I. Preprocessing flow

### 1. Extra steps

OpenLane annotations provide lane lines as sequences of (u, v) coordinates, with each lane potentially containing a large number of points. To ensure consistency and efficiency, the following steps are performed:

- **Sampling:** : lanes with excessive points are downsampled to a manageable number using a configurable threshold.
- **Sorting:** : lane points are sorted by their y-coordinate (vertical axis) to maintain a consistent bottom-to-top order.
- **Deduplication:** : adjacent points with identical y-coordinates are filtered out to avoid redundancy.
- **Anchor calculation:** : each lane is assigned an anchor point at the bottom of the image, along with linear fit parameters for further processing.
- **Lane classification:** : lanes are classified as left ego, right ego, or other, based on their anchor positions and attributes.
- **Drivable path generation:** : the drivable path is computed as the midpoint between the left and right ego lanes.

### 2. Technical implementations

Most functions accept parameters for controlling the number of points per lane (`lane_point_threshold`) and verbosity for debugging. All coordinates are rounded to two decimal places for efficiency and are not normalized until just before saving to JSON.

During processing, each image and its associated lanes are handled with careful attention to coordinate consistency, especially when resizing or cropping is involved in downstream tasks.


## II. Usage

### 1. Args

- `--dataset_dir` : path to the OpenLane dataset directory, which should contain the raw JSON annotation files and images.
- `--output_dir` : path to the directory where processed images and annotations will be saved.
- `--sampling_step` : (optional) specifies the interval for sampling images (e.g., process 1 image, skip 4). Default is 5.
- `--early_stopping` : (optional) for debugging; stops processing after a specified number of images.

### 2. Execute

```bash
# Example: process first 100 images with default sampling
python3 EgoPath/create_path/OpenLane/process_openlane.py --dataset_dir ../pov_datasets/OpenLane --output_dir ../pov_datasets/OpenLane_Processed --sampling_step 5 --early_stopping 100
```


## III. Functions

### 1. `parseData(json_data, lane_point_threshold=20, verbose=False)`
- **Description**: parses a single OpenLane annotation entry, extracting and processing lane lines, classifying ego lanes, and generating the drivable path.
- **Parameters**:
    - `json_data` (dict): raw annotation data for one image.
    - `lane_point_threshold` (int): maximum number of points per lane.
    - `verbose` (bool): enables detailed warnings and debug info.
- **Returns**: a dictionary with processed lanes, ego lanes, and drivable path.

### 2. `normalizeCoords(lane, width, height)`
- **Description**: Normalizes lane coordinates to the `[0, 1]` range based on image dimensions.
- **Parameters**:
    - `lane` (list of tuples): List of `(x, y)` points.
    - `width` (int): Image width.
    - `height` (int): Image height.
- **Returns**: List of normalized `(x, y)` points.

### 3. `getLineAnchor(lane, img_height)`
- **Description**: Computes the anchor point and linear fit parameters for a lane at the bottom of the image.
- **Parameters**:
    - `lane` (list of tuples): Lane points.
    - `img_height` (int): Image height.
- **Returns**: Tuple `(x0, a, b)` for anchor and line fit.

### 4. `getEgoIndexes(anchors, img_width)`
- **Description**: Identifies the left and right ego lanes from sorted anchors.
- **Parameters**:
    - `anchors` (list of tuples): Lane anchors sorted by x-coordinate.
    - `img_width` (int): Image width.
- **Returns**: Tuple `(left_ego_idx, right_ego_idx)`.

### 5. `getDrivablePath(left_ego, right_ego, img_height, img_width, y_coords_interp=False)`
- **Description**: Computes the drivable path as the midpoint between left and right ego lanes.
- **Parameters**:
    - `left_ego` (list of tuples): Left ego lane points.
    - `right_ego` (list of tuples): Right ego lane points.
    - `img_height` (int): Image height.
    - `img_width` (int): Image width.
    - `y_coords_interp` (bool): Whether to interpolate y-coordinates for smoother curves.
- **Returns**: List of `(x, y)` points for the drivable path.

### 6. `annotateGT(raw_img, anno_entry, raw_dir, visualization_dir, mask_dir, img_width, img_height, normalized=True)`
- **Description**: Annotates and saves an image with lane markings, drivable path, and segmentation mask.
- **Parameters**:
    - `raw_img` (PIL.Image): Original image.
    - `anno_entry` (dict): Processed annotation data.
    - `raw_dir` (str): Directory for raw images.
    - `visualization_dir` (str): Directory for annotated images.
    - `mask_dir` (str): Directory for segmentation masks.
    - `img_width` (int): Image width.
    - `img_height` (int): Image height.
    - `normalized` (bool): Whether coordinates are normalized.
- **Returns**: None