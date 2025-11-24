# OpenLane Dataset Processing Script


## Overview

OpenLane is a large-scale benchmark for lane detection and topology estimation, widely used in
autonomous driving and ADAS research. The dataset features diverse road scenarios, complex
lane topologies, and high-resolution images. This script suite provides tools for parsing,
preprocessing, and transforming OpenLane data for downstream tasks such as drivable path
detection, BEV (Bird-Eyes-View) transformation, and model training.


## I. Preprocessing flow

### 1. Extra steps

OpenLane annotations provide lane lines as sequences of (u, v) coordinates, with each lane
potentially containing a large number of points. To ensure consistency and efficiency, the
following steps are performed:

- **Sampling:** : lanes with excessive points are downsampled to a manageable number using
a configurable threshold.
- **Sorting:** : lane points are sorted by their y-coordinate (vertical axis) to maintain
a consistent bottom-to-top order.
- **Deduplication:** : adjacent points with identical y-coordinates are filtered out to
avoid redundancy.
- **Anchor calculation:** : each lane is assigned an anchor point at the bottom of the
image, along with linear fit parameters for further processing.
- **Lane classification:** : lanes are classified as left ego, right ego, or other, based
on their anchor positions and attributes.
- **Drivable path generation:** : the drivable path is computed as the midpoint between
the left and right ego lanes.

### 2. Technical implementations

Most functions accept parameters for controlling the number of points per lane
(`lane_point_threshold`) and verbosity for debugging. All coordinates are rounded to two
decimal places for efficiency and are not normalized until just before saving to JSON.

During processing, each image and its associated lanes are handled with careful attention
to coordinate consistency, especially when resizing or cropping is involved in downstream tasks.


## II. Usage

### 1. Args

- `--dataset_dir` : path to the OpenLane dataset directory, which should contain the raw
JSON annotation files and images.
- `--output_dir` : path to the directory where processed images and annotations will be saved.
- `--sampling_step` : (optional) specifies the interval for sampling images (e.g., process
1 image, skip 4). Default is 5.
- `--early_stopping` : (optional) for debugging; stops processing after a specified number
of images.

### 2. Execute

```bash
# Example: process first 100 images with default sampling
python3 EgoPath/create_path/OpenLane/process_openlane.py --dataset_dir ../pov_datasets/OpenLane --output_dir ../pov_datasets/OpenLane_Processed --sampling_step 5 --early_stopping 100
```


## III. Functions

### 1. `parseData(json_data, lane_point_threshold=20, verbose=False)`
- **Description**: parses a single OpenLane annotation entry, extracting and processing lane
lines, classifying ego lanes, and generating the drivable path.
- **Parameters**:
    - `json_data` (dict): raw annotation data for one image.
    - `lane_point_threshold` (int): maximum number of points per lane.
    - `verbose` (bool): enables detailed warnings and debug info.
- **Returns**: a dictionary with processed lanes, ego lanes, and drivable path.

### 2. `normalizeCoords(lane, width, height)`
- **Description**: normalizes lane coordinates to the `[0, 1]` range based on image dimensions.
- **Parameters**:
    - `lane` (list of tuples): list of `(x, y)` points.
    - `width` (int): image width.
    - `height` (int): image height.
- **Returns**: list of normalized `(x, y)` points.

### 3. `getLineAnchor(lane, img_height)`
- **Description**: computes the anchor point and linear fit parameters for a lane at the bottom
of the image.
- **Parameters**:
    - `lane` (list of tuples): lane points.
    - `img_height` (int): image height.
- **Returns**: tuple `(x0, a, b)` for anchor and line fit.

### 4. `getEgoIndexes(anchors, img_width)`
- **Description**: identifies the left and right ego lanes from sorted anchors.
- **Parameters**:
    - `anchors` (list of tuples): lane anchors sorted by x-coordinate.
    - `img_width` (int): image width.
- **Returns**: tuple `(left_ego_idx, right_ego_idx)`.

### 5. `getDrivablePath(left_ego, right_ego, img_height, img_width, y_coords_interp=False)`
- **Description**: Computes the drivable path as the midpoint between left and right ego lanes.
- **Parameters**:
    - `left_ego` (list of tuples): left ego lane points.
    - `right_ego` (list of tuples): right ego lane points.
    - `img_height` (int): image height.
    - `img_width` (int): image width.
    - `y_coords_interp` (bool): whether to interpolate y-coordinates for smoother curves.
- **Returns**: list of `(x, y)` points for the drivable path.

### 6. `annotateGT(raw_img, anno_entry, raw_dir, visualization_dir, mask_dir, img_width, img_height, normalized=True)`
- **Description**: annotates and saves an image with lane markings, drivable path, and segmentation mask.
- **Parameters**:
    - `raw_img` (PIL.Image): original image.
    - `anno_entry` (dict): processed annotation data.
    - `raw_dir` (str): directory for raw images.
    - `visualization_dir` (str): directory for annotated images.
    - `mask_dir` (str): directory for segmentation masks.
    - `img_width` (int): image width.
    - `img_height` (int): image height.
    - `normalized` (bool): whether coordinates are normalized.
- **Returns**: none


# OpenLane Dataset BEV (Bird-Eye-View) Processing Script

## Overview

This script processes OpenLane dataset frames to generate BEV representations of the drivable path.
It reads per-frame image and lane data, computes the BEV transform using ego-lane boundaries,
projects the drivable path into BEV space, and saves both the transformed images and path data for
downstream use (training, visualization, etc.).

## I. Algorithm

### From EgoPath dataset

For each frame, the following ground-truth information is available:
- Left and right ego lanes.
- Drivable path (algorithmically derived).
- Other lanes (not used for BEV transformation).

To perform BEV transformation, four frustum points are required:
- Left start (`LS`)
- Right start (`RS`)
- Left end (`LE`)
- Right end (`RE`)

### Step 1

Obtain anchors for left and right ego lanes:
- Left anchor: $(x_{0L}, a_L, b_L)$ at $(x_{0L}, h)$
- Right anchor: $(x_{0R}, a_R, b_R)$ at $(x_{0R}, h)$

These serve as `LS` and `RS`.

### Step 2

- Compute the midpoint `MS` at $x_{M_S} = (x_{0L} + x_{0R}) / 2$.
- Compute average tangent $a_M$ at `MS`.

### Step 3

- Extend from `MS` along $a_M$ to the height of the shorter ego lane (`lower_ego_y`), yielding `ME`.

### Step 4

- Calculate the width between ego lanes at $y = lower_ego_y$, denoted as $𝚫D = 2𝚫d$.

### Step 5

- At $y = lower_ego_y$, compute $x = x_{M_E} ± 𝚫d$ for `LE` and `RE`.

### Step 6

With `LS`, `RS`, `LE`, and `RE` defined, perform the BEV transformation using a homography.


### 2. Execute

```bash
# Example: process first 100 images for BEV transformation
python3 EgoPath/create_path/OpenLane/parse_openlane_bev.py --dataset_dir ../pov_datasets/OpenLane_Processed --early_stopping 100
```

## III. Functions

### 1. `drawLine(img, line, color, thickness=2)`

- **Description**: Draws a polyline on an image given a list of points.
- **Parameters**:
    - `img` (np.ndarray): Image to draw on.
    - `line` (list): List of `(x, y)` tuples.
    - `color` (tuple): BGR color.
    - `thickness` (int): Line thickness.
- **Returns**: None

### 2. `annotateGT(img, frame_id, bev_egopath, raw_dir, visualization_dir, normalized)`

- **Description**: Annotates and saves BEV images with the drivable path.
- **Parameters**:
    - `img` (np.ndarray): BEV image.
    - `frame_id` (str): Frame identifier.
    - `bev_egopath` (list): Drivable path in BEV space.
    - `raw_dir` (str): Directory for raw BEV images.
    - `visualization_dir` (str): Directory for BEV visualizations.
    - `normalized` (bool): Whether coordinates are normalized.
- **Returns**: None

### 3. `interpX(line, y)`

- **Description**: Interpolates the x-value of a line at a given y-coordinate.
- **Parameters**:
    - `line` (list): List of `(x, y)` points.
    - `y` (float): Target y-coordinate.
- **Returns**: Interpolated x-value.

### 4. `polyfit_BEV(bev_egopath, order, y_step, y_limit)`

- **Description**: Fits a polynomial to the BEV path and generates evenly spaced points.
- **Parameters**:
    - `bev_egopath` (list): BEV path points.
    - `order` (int): Polynomial order (usually 2).
    - `y_step` (int): Step size for y.
    - `y_limit` (int): Maximum y for interpolation.
- **Returns**:
    - `fitted_bev_egopath` (tuple): Fitted `(x, y)` points.
    - `flag_list` (list): Validity flags for each x.

### 5. `imagePointTuplize(point)`

- **Description**: Converts a point to integer coordinates for image operations.
- **Parameters**:
    - `point` (tuple): `(float, float)` point.
- **Returns**: `(int, int)` tuple.

### 6. `findSourcePointsBEV(h, w, egoleft, egoright)`

- **Description**: Computes four source points for BEV homography.
- **Parameters**:
    - `h` (int): Image height.
    - `w` (int): Image width.
    - `egoleft` (list): Left ego lane points.
    - `egoright` (list): Right ego lane points.
- **Returns**: Dict with `LS`, `RS`, `LE`, `RE`, and `ego_h`.

### 7. `transformBEV(img, egopath, sps)`

- **Description**: Applies perspective transform to image and path for BEV.
- **Parameters**:
    - `img` (np.ndarray): Original image.
    - `egopath` (list): Drivable path.
    - `sps` (dict): Source points for homography.
- **Returns**:
    - `im_dst` (np.ndarray): BEV image.
    - `bev_egopath` (list): Polyfitted BEV path.
    - `flag_list` (list): Validity flags.
    - `mat` (np.ndarray): Homography matrix.


## IV. Running All at Once

```bash
python3 EgoPath/create_path/OpenLane/process_openlane.py --dataset_dir ../pov_datasets/OpenLane --output_dir ../pov_datasets/OpenLane_Processed
python3 EgoPath/create_path/OpenLane/parse_openlane_bev.py --dataset_dir ../pov_datasets/OpenLane_Processed
```