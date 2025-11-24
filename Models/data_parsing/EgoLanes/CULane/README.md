# CULane Dataset Processing Script

## Overview

CULane is a large-scale lane detection dataset specifically designed for autonomous driving research. It consists of images captured from a car-mounted camera moving in Beijing, China, under various real-world scenarios, including urban traffic, highways, and challenging weather conditions. This script processes the CULane dataset to generate normalized annotations, drivable paths, and segmentation masks for lane detection tasks. Basically, these scripts produce:

- Normalized lane annotations
- Ego lane identification (left and right)
- Drivable path computation
- Visualization outputs
- Polyfitted BEV projections
- Reprojected lane visualizations

![alt text](../../../Media/CULane_dataset_preview.png)

---

## 1. CULane raw dataset processing (`process_culane.py`)

### Functions

#### `normalizeCoords(lane, width, height)`
- **Description**: normalizes lane coordinates to a scale of `[0, 1]` based on image dimensions.
- **Parameters**:
  - `lane` (list of tuples): list of `(x, y)` points defining a lane.
  - `width` (int): width of the image.
  - `height` (int): height of the image.
- **Returns**: a list of normalized `(x, y)` points.


#### `getLaneAnchor(lane)`
- **Description**: determines the "anchor" point of a lane, which is used to identify ego lanes.
- **Parameters**:
  - `lane` (list of tuples): list of `(x, y)` points defining a lane.
- **Returns**: a tuple containing the anchor `(x0, a, b)` for the lane, where `x0` is the x-coordinate at the bottom of the frame, and `(a, b)` are slope and intercept of the lane equation.


#### `getEgoIndexes(anchors)`
- **Description**: identifies the left and right ego lanes from a list of lane anchors.
- **Parameters**:
  - `anchors` (list of tuples): list of lane anchors.
- **Returns**: a tuple `(left_ego_idx, right_ego_idx)` of the indexes of the two ego lanes, or a warning message if insufficient lanes are detected.


#### `getDrivablePath(left_ego, right_ego)`
- **Description**: computes the drivable path as the midpoint between the two ego lanes.
- **Parameters**:
  - `left_ego` (list of tuples): points defining the left ego lane.
  - `right_ego` (list of tuples): points defining the right ego lane.
- **Returns**: a list of `(x, y)` points representing the drivable path.


#### `annotateGT(anno_entry, anno_raw_file, raw_dir, visualization_dir, mask_dir, img_width, img_height, normalized=True, crop=None)`
- **Description**: annotates and saves images with lanes, drivable paths, and binary masks.
- **Parameters**:
  - `anno_entry` (dict): normalized annotation data.
  - `anno_raw_file` (str): path to the raw image file.
  - `raw_dir` (str): directory for saving raw images.
  - `visualization_dir` (str): directory for saving annotated images.
  - `img_width` (int): width of the processed image.
  - `img_height` (int): height of the processed image.
  - `normalized` (bool): if `True`, annotations are normalized.
  - `crop` (dict): Crop dimensions in the format `{"TOP": int, "RIGHT": int, "BOTTOM": int, "LEFT": int}`.
- **Returns**: none.


#### `parseAnnotations(anno_path, crop=None)`
- **Description**: parses lane annotations and extracts normalized ground truth data.
- **Parameters**:
  - `anno_path` (str): path to the annotation file.
  - `crop` (dict): crop dimensions in the format `{"TOP": int, "RIGHT": int, "BOTTOM": int, "LEFT": int}`.
- **Returns**: a dictionary containing normalized lanes, ego indexes, and drivable path, or `None` if parsing fails.

---

### Usage

#### Args
- `--dataset_dir`: path to the CULane dataset directory.
- `--output_dir`: path to the directory where processed outputs will be saved.
- `--crop`: optional. Crop dimensions as `[TOP, RIGHT, BOTTOM, LEFT]`. Default is `[0, 390, 160, 390]`.
- `--sampling_step`: optional. Sampling step for each split/class. Default is `5`.
- `--early_stopping`: optional. Stops after processing a specific number of files for debugging purposes.

---

### Outputs

```bash
--output_dir
    ├── image/                # Cropped raw images
    ├── visualization/        # Raw image with lane/path overlays
    └── drivable_path.json    # Normalized annotation data
```

---

### Running the script

```bash
python process_culane.py --dataset_dir /path/to/CULane --output_dir /path/to/output
```

---

## 2. CULane raw dataset processing (`parse_culane_bev.py`)

### Functions


#### `log_skipped(frame_id, reason)`

- **Description**: Records skipped frame IDs and reasons into a global dictionary for later reporting.
- **Parameters**:
  - `frame_id` (str): identifier of the frame being skipped.
  - `reason` (str): explanation for skipping the frame.
- **Returns**: None


#### `imagePointTuplize(point)`

- **Description**: converts a floating-point `(x, y)` coordinate into integer pixel coordinates for image drawing or indexing.
- **Parameters**:
  - `point` (tuple): floating-point coordinate `(x, y)`.
- **Returns**: integer coordinate `(x, y)`.


#### `roundLineFloats(line, ndigits=4)`

- **Description**: rounds each point in a line to a given number of decimal places.
- **Parameters**:
  - `line` (list): list of `(x, y)` tuples.
  - `ndigits` (int): number of decimal places to round to. Default is 4.
- **Returns**: rounded list of `(x, y)` tuples.


#### `normalizeCoords(line, width, height)`

- **Description**: normalizes coordinates to the `[0, 1]` range using image dimensions.
- **Parameters**:
  - `line` (list): list of `(x, y)` tuples.
  - `width` (int): image width.
  - `height` (int): image height.
- **Returns**: normalized list of `(x, y)` tuples.


#### `interpLine(line, points_quota)`

- **Description**: interpolates additional points along a line based on arc length to meet a minimum quota.
- **Parameters**:
  - `line` (list): list of `(x, y)` tuples.
  - `points_quota` (int): required minimum number of points.
- **Returns**: interpolated list of `(x, y)` tuples.


#### `getLineAnchor(line)`

- **Description**: computes an anchor `(x0, a, b)` representing the lane's intercept and slope at the bottom of the image. Warns on vertical or horizontal lanes.
- **Parameters**:
  - `line` (list): list of `(x, y)` tuples.
- **Returns**: tuple `(x0, a, b)` where `x0` is the intercept, and `a`, `b` are slope and intercept.


#### `drawLine(img, line, color, thickness=2)`

- **Description**: draws a polyline on an image using OpenCV.
- **Parameters**:
  - `img` (ndarray): target image.
  - `line` (list): list of `(x, y)` tuples.
  - `color` (tuple): BGR color.
  - `thickness` (int): line thickness. Default is 2.
- **Returns**: none


#### `annotateGT(img, orig_img, frame_id, bev_egopath, reproj_egopath, bev_egoleft, reproj_egoleft, bev_egoright, reproj_egoright, raw_dir, visualization_dir, normalized)`

- **Description**: annotates and saves both BEV-space and original-space visualizations of ego lanes and drivable path.
- **Parameters**:
  - `img` (ndarray): BEV image.
  - `orig_img` (ndarray): original image.
  - `frame_id` (str): image ID used for saving.
  - `bev_egopath`, `bev_egoleft`, `bev_egoright` (list): BEV-space lines.
  - `reproj_egopath`, `reproj_egoleft`, `reproj_egoright` (list): reprojected lines.
  - `raw_dir` (str): directory for BEV raw images.
  - `visualization_dir` (str): directory for visualizations.
  - `normalized` (bool): whether coordinates are normalized.
- **Returns**: none


#### `interpX(line, y)`

- **Description**: interpolates the x-coordinate along a lane line at a given y-value.
- **Parameters**:
  - `line` (list): list of `(x, y)` tuples.
  - `y` (float): target y-value.
- **Returns**: interpolated x-value (float).


#### `polyfit_BEV(bev_line, order, y_step, y_limit)`

- **Description**: fits a polynomial to a BEV lane and samples points at regular y-intervals.
- **Parameters**:
  - `bev_line` (list): list of `(x, y)` tuples in BEV space.
  - `order` (int): polynomial degree.
  - `y_step` (int): y-interval for sampling.
  - `y_limit` (int): max y-value.
- **Returns**: tuple `(fitted_line, flag_list, validity_list)`.


#### `findSourcePointsBEV(h, w, egoleft, egoright)`

- **Description**: computes the 4 source points for homography (left/right ego start/end) and ego bottom height.
- **Parameters**:
  - `h` (int): image height.
  - `w` (int): image width.
  - `egoleft` (list): left ego lane (normalized).
  - `egoright` (list): right ego lane (normalized).
- **Returns**: dictionary of source points and ego height.


#### `transformBEV(img, line, sps)`

- **Description**: transforms a line to BEV space, fits a polyline, and reprojects it back to the original space.
- **Parameters**:
  - `img` (ndarray): input image.
  - `line` (list): normalized line to transform.
  - `sps` (dict): source points for homography.
- **Returns**: tuple:
  - `im_dst`: warped BEV image,
  - `bev_line`: fitted BEV line,
  - `reproj_line`: reprojected original-space line,
  - `flag_list`: polyfit flags,
  - `validity_list`: point validity,
  - `mat`: homography matrix,
  - `success`: transformation success flag.

---

### Usage

#### Outputs

```bash
--output_dir
    ├── image_bev/            # BEV-transformed images
    ├── visualization_bev/    # BEV + reprojected visualization
    ├── drivable_path_bev.json   # Polyfitted BEV + reprojected annotations
    └── skipped_frames.json      # Log of skipped frames with reasons
```

#### Args

- `--dataset_dir` (required)
  - Path to the processed CULane dataset directory (i.e., the output directory from `process_culane.py`).
  - Example : `../pov_datasets/CULANE`

- `--early_stopping` (optional)
  - Limits the number of frames processed. Useful for debugging or quick testing, default `None` (processes all frames).
  - Example : `--early_stopping 100`

#### Running the script

Once you have processed the CULane dataset using `process_culane.py`, you can run the BEV transformation script to generate BEV views and reprojected visualizations.

```bash
python3 EgoPath/create_path/CULane/parse_culane_bev.py \
  --dataset_dir ../pov_datasets/CULANE
```

You can limit the number of frames for debugging or development purposes. This will process first 100 frames.

```bash
python3 EgoPath/create_path/CULane/parse_culane_bev.py \
  --dataset_dir ../pov_datasets/CULANE \
  --early_stopping 100
```

## 3. End-to-end run

```bash
python3 EgoPath/create_path/CULane/process_culane.py --dataset_dir ../pov_datasets/CULane --output_dir ../pov_datasets/CULANE
python3 EgoPath/create_path/CULane/parse_culane_bev.py --dataset_dir ../pov_datasets/CULANE/
```