# CurveLane Dataset Processing Script

## Overview

CurveLane is a large-scale lane detection dataset specialized in various AD/ADAS-related tasks, including lane detection, drivable path detection, object detection, path planning & decision, etc. The key feature of CurveLane, is the abundant presence of extremely curved path sections, which might take up to 40% of the 100k-image dataset. The majority of CurveLane images are in huge resolution, typically 2k and none of them are smaller than 1280x720.

## I. Preprocessing flow

### 1. Extra steps

CurveLane has the same vertical direction of labeling as CULane, which is lower to upper, but quite random for horizontal direction (can be any lane), which must be sorted left to right so we can reuse the CULane pipeline with ease. After that, most of the functions stay the same as CULane, except the extra processes of resizing and cropping I added in order to express the problem of the images being way too huge compared to our needs.

According to my EDA, across 100k images in CurveLanes we observe 3 different sizes:

- `2560 x 1440` : 109809 images
- `1570 x 660` : 10180 images
- `1280 x 720` : 11 images

As we agree during earlier meeting (not sure when exactly), we will convert all of em to `800 x 400`. Thus, 
the flow is as follow:

- `2560 x 1440` ==|`resize(0.5)`|==> `1280 x 720` ==|`crop(240, 160, 240, 160)`|==> `800 x 400`
- `1570 x 660` ==|`crop(130, 385, 130, 385)`|==> `800 x 400`
- `1280 x 720` ==|`crop(240, 160, 240, 160)`|==> `800 x 400`

### 2. Technical implementations

Now, most of the functions will be accompanied by 2 extra params:

- `resize (float)` : indicates resizing ratio of each image. As per the flow proposed above, our typical resizing ratio is `0.5`.
- `crop (1x4 int tuple)` : indicates crop width, being top, right, bottom, left size respectively. A crop tuple of `(a, b, c, d)` means that image will be cropped `a` pixels at top size, `b` pixels at right size, `c` pixels at bottom size, and `d` pixels at left size. Basically this order follows the one used in HTML, so I just apply it here.

Each image, upon being parsed into those functions, will have 2 params of `original_img_width/height` and `new_img_width/height`. Initially the `new` is declared the same as `original`. Each time a resize/crop action is called, the `new` is updated accordingly. This is to ensure the image's coordinates and annotations being consistently across the processing flow, since our legacy code heavily relies on the current image's size to do its biddings (this is also a pain in my ass as well, so I hope we could do some refactoring later, if anyone is interested).

## II. Usage

### 1. Args

- `--dataset_dir` : path to CurveLane dataset directory, should contains exactly `Curvelanes` if you get it from Kaggle.
- `--output_dir` : path to directory where you wanna save the images.
- `--sampling_step` : optional. Basically tells the process to skip several images for increased model learning capability during the latter training. Default is 5, which means process 1 image then skip 4, and continue.
- `--early_stopping` : optional. For debugging purpose. Force the process to halt upon reaching a certain amount of images.

## 2. Execute

```bash
# `pov_datasets` includes `Curvelanes` directory in this case
# Sampling rate 5 (also by default)
# Process first 100 images, then stop (for a quick run, instead of going through 150k images)
python3 EgoPath/create_path/CurveLanes/process_curvelanes.py --dataset_dir ../pov_datasets --output_dir ../pov_datasets/Output --sampling_step 5 --early_stopping 100
```

## III. Functions

### 1. `normalizeCoords(lane, width, height)`
- **Description**: normalizes lane coordinates to a scale of `[0, 1]` based on image dimensions.
- **Parameters**:
    - `lane` (list of tuples): list of `(x, y)` points defining a lane.
    - `width` (int): width of the image.
    - `height` (int): height of the image.
- **Returns**: a list of normalized `(x, y)` points.

### 2. `getLaneAnchor(lane, new_img_height)`
- **Description**: determines the "anchor" point of a lane, which helps in lane classification.
- **Parameters**:
    - `lane` (list of tuples): list of `(x, y)` points representing a lane.
    - `new_img_height` (int): height of the resized or cropped image.
- **Returns**: a tuple `(x0, a, b)` representing the x-coordinate of the anchor and the linear equation parameters `a` and `b` for the lane.

### 3. `getEgoIndexes(anchors, new_img_width)`
- **Description**: identifies two ego lanes (left and right) from a sorted list of lane anchors.
- **Parameters**:
    - `anchors` (list of tuples): list of lane anchors sorted by x-coordinate.
    - `new_img_width` (int): width of the processed image.
- **Returns**: a tuple `(left_ego_idx, right_ego_idx)` with the indexes of the two ego lanes or an error message if lanes are missing.

### 4. `getDrivablePath(left_ego, right_ego, new_img_height, new_img_width, y_coords_interp=False)`
- **Description**: computes the drivable path as the midpoint between two ego lanes.
- **Parameters**:
    - `left_ego` (list of tuples): points of the left ego lane.
    - `right_ego` (list of tuples): points of the right ego lane.
    - `new_img_height` (int): height of the processed image.
    - `new_img_width` (int): width of the processed image.
    - `y_coords_interp` (bool, optional): whether to interpolate y-coordinates for smoother curves. Defaults to `False`.
- **Returns**: a list of `(x, y)` points representing the drivable path, or an error message if the path violates heuristics.

### 5. `annotateGT(raw_img, anno_entry, raw_dir, visualization_dir, mask_dir, init_img_width, init_img_height, normalized=True, resize=None, crop=None)`
- **Description**: annotates and saves an image with lane markings, drivable path, and segmentation mask.
- **Parameters**:
    - `raw_img` (PIL.Image): the original image.
    - `anno_entry` (dict): annotation data including lanes and drivable path.
    - `raw_dir` (str): directory to save raw images.
    - `visualization_dir` (str): directory to save annotated images.
    - `mask_dir` (str): directory to save segmentation masks.
    - `init_img_width` (int): original width of the image.
    - `init_img_height` (int): original height of the image.
    - `normalized` (bool, optional): whether coordinates are normalized. Defaults to `True`.
    - `resize` (float, optional): resize factor. Defaults to `None`.
    - `crop` (dict, optional): cropping dimensions. Defaults to `None`.
- **Returns**: None

# CurveLane Dataset BEV (Bird-Eyes-View) Processing Script

## Overview

This script processes annotated CurveLanes dataset frames to generate a bird-eyes view (BEV) representation of the drivable path. It reads per-frame image and lane data, computes the BEV transform using ego-lane boundaries, projects the drivable path into BEV space, and saves both the transformed images and path data for downstream use (training, visualization, etc.).

## I. Algorithm

### From EgoPath dataset

From EgoPath datasets, each frame we have ground-truth info of:
- Egolines, left ego line and right ego line.
- Drivable path, derived from algorithm.
- Other lanes, but for now no need to care about them.

In order to conduct BEV transformation, we need 4 points of a frustum:
- Left start (`LS`)
- Right start (`RS`)
- Left end (`LE`)
- Right end (`RE`)

### Step 1

Acquire 2 anchors of left and right egolines:
- Left anchor : $(x_{0L}, a_L, b_L)$ with anchor point $(x_{0L}, h)$
- Right anchor : $(x_{0R}, a_R, b_R)$ with anchor point $(x_{0R}, h)$

In which:
- $x0$ : intercept of egoline and bottom edge of frame
- `a, b` : linear constants of that egoline, at bottom edge.

These 2 points will serve as left start and right start of the frustum - `LS` and `RS` - respectively.

### Step 2

- Get midpoint of 2 egolines `MS` (mid-start) at $x_{M_S} = (x_{0L} + x_{0R}) / 2$.
- Get a tangential value $a_M$ at `MS` that is average of the 2 values $a_L$ and $a_R$.

### Step 3

- Extend from midpoint along $a_M$ until it intercepts the height of shorter egoline, denoted by `lower_ego_y`.
- This new intercept point is denoted by `ME` (mid-end).

### Step 4

- Calculate the width between left and right egolines at $y_L = y_R = $ `lower_ego_y`.
- This width is denoted by $𝚫D = 2𝚫d$

### Step 5

- At $y = $ `lower_ego_y`, obtain 2 points at $x = x_{M_E} ± 𝚫d$ (remember $𝚫D = 2𝚫d$).
- These 2 points will serve as left end `LE` and right end `RE` of the frustum.

### Step 6

Now the frustum is completed, we can conduct BEV transformation based on the 4 source points left start (`LS`), right start (`RS`), left end (`LE`) and right end (`RE`).

## II. Usage

### 1. Args

- `--dataset_dir` : path to **processed** CurveLane dataset directory, including `image`, `visualization` and `drivable_path.json`.
- `--early_stopping` : optional. For debugging purpose. Force the process to halt upon reaching a certain amount of images.

## 2. Execute

```bash
# `pov_datasets` includes `CURVELANES` which is the processed CurveLanes directory in this case
# Process first 100 images, then stop (for a quick run, instead of going through ~100k processed images)
python3 EgoPath/create_path/CurveLanes/parse_curvelanes_bev.py --dataset_dir ../pov_datasets/CURVELANES --early_stopping 100
```

## III. Functions

### 1. `drawLine(img, line, color, thickness=2)`

- **Description**: draws a polyline on an image given a list of points.
- **Parameters**:
    - `img` (np.ndarray): the image on which to draw the line.
    - `line` (list): a list of `(x, y)` tuples representing the line points.
    - `color` (tuple): the BGR color of the line.
    - `thickness` (int, optional): thickness of the line. Defaults to `2`.
- **Returns**: None

### 2. `annotateGT(img, frame_id, bev_egopath, raw_dir, visualization_dir, normalized)`

- **Description**: annotates and saves both raw and visualization images with the drivable path.
- **Parameters**:
    - `img` (np.ndarray): the BEV-transformed image.
    - `frame_id` (str): identifier for current frame.
    - `bev_egopath` (list): drivable path line in BEV space.
    - `raw_dir` (str): directory to save the raw BEV image.
    - `visualization_dir` (str): directory to save the BEV visualization image.
    - `normalized` (bool): whether the coords are normalized to `[0, 1]`.
- **Returns**: None

### 3. `interpX(line, y)`

- **Description**: interpolates the x-value of a line at a given y-coordinate.
- **Parameters**:
    - `line` (list): a list of `(x, y)` tuples representing the line points.
    - `y` (float): the y-value at which to interpolate the x-value.
- **Returns**: the interpolated x-value as a `float`.

### 4. `polyfit_BEV(bev_egopath, order, y_step, y_limit)`

- **Description**: fits a polynomial curve to the BEV path and generates evenly spaced points along the y-axis.
- **Parameters**:
    - `bev_egopath` (list): list of `(x, y)` points in BEV space.
    - `order` (int): order of the polynomial fit, preferrably `2`.
    - `y_step` (int): step size between y-values.
    - `y_limit` (int): maximum y-value for interpolation.
- **Returns**:
    - `fitted_bev_egopath` (tuple): tuple of fitted `(x, y)` points.
    - `flag_list` (list): list of booleans indicating whether each x is within valid bounds.

### 5. `imagePointTuplize(point)`

- **Description**: converts a point's coordinates to integers for image operations.
- **Parameters**:
    - `point` (tuple): a `(float, float)` point to be converted.
- **Returns**: an `(int, int)` tuple of the converted point.

### 6. `findSourcePointsBEV(h, w, egoleft, egoright)`

- **Description**: computes four source points for homography transformation to BEV space.
- **Parameters**:
    - `h` (int): height of the image.
    - `w` (int): width of the image.
    - `egoleft` (list): normalized points representing the left ego lane.
    - `egoright` (list): normalized points representing the right ego lane.
- **Returns**: a dictionary with keys `LS`, `RS`, `LE`, `RE` for source points, and `ego_h` for ego lane height.

### 7. `transformBEV(img, egopath, sps)`

- **Description**: applies a perspective transform to convert the image and drivable path into BEV space.
- **Parameters**:
    - `img` (np.ndarray): original perspective-view image.
    - `egopath` (list): drivable path in normalized coordinates.
    - `sps` (dict): source points for the homography transform.
- **Returns**:
    - `im_dst` (np.ndarray): transformed BEV image.
    - `bev_egopath` (list): polyfitted drivable path in BEV space.
    - `flag_list` (list): boolean list indicating valid x-values in BEV space.
    - `mat` (np.ndarray): homography matrix used for the transformation.

## IV. Running all at once

```
python3 EgoPath/create_path/CurveLanes/process_curvelanes.py --dataset_dir ../pov_datasets/ --output_dir ../pov_datasets/CurveLanes_250624
python3 EgoPath/create_path/CurveLanes/parse_curvelanes_bev.py --dataset_dir ../pov_datasets/CurveLanes_250624
```