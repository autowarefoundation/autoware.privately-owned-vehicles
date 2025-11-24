# process_tusimple.py

## TuSimple dataset preprocessing script for PathDet.

This script parse the [TuSimple lane detection dataset](https://www.kaggle.com/datasets/manideep1108/tusimple?resource=download) (24GB) to create a dataset comprising input images in PNG format and a single drivable path as the ground truth, derived as the mid-line between the left/right ego lanes.

- Data acquisition: by TuSimple - an autonomous trucking company, with 6,408 road images on US highways.
- Features:
    - Different conditions (weather, light, highway, traffic, etc.)
    - Lane detection competition in CVPR 2017 WAD.
- Annotation method: polylines for lane markings

- TuSimple directory structure (from download)
    - `clips/` : video clips
    - `some_clip/` : sequential images, 20 frames
    - `tasks.json` : label data in training set, and a submission template for testing set.

- Data label format:
    - `raw_file` : (str) 20th frame file path in a clip
    - `lanes` : (list) lists of lanes, each list represents a lane. Each element is X-coordinate across the polyline.
    - `h_samples` : (list) list of Y-coordinate across the polyline.
        - `-2` means there is no existing lane marking.
    - Normally each frame has 4 lanes (ego x 2, left, right), but some has 5 (changing lane).



## I. Functions

### 1. `normalizeCoords()`

Normalize the coords of lane points.

#### a. Parameters
- `lane` (list of tuples):
    - list of (x, y) tuples representing 2D coords of lane points.
    - from here please be reminded that all these `lane` used in this script are in ascending order of y-coords, which means it starts from top to bottom.
- `width` (float): 
    - image width, 1280 for TuSimple.
- `height` (float):
    - image height, 720 for TuSimple.

#### b. Returns
- normalized lane: 
    - list of (x, y) tuples with normalized coords.

### 2. `getLaneAnchor()`

Determine "anchor" point of a lane.

Here I define, the *anchor* of a lane is the intersection point of a lane with the bottom edge of an image, determined by the lane's linear equation, defined by its 2 points:
- `(x2, y2)`: last point of the lane, closest to bottom edge (where `y = img_height = 720`).
- `(x1, y1)`: closest point to `(x2, y2)` but with different x-coord.

With these 2 points, slope `a` and y-intercept `b` of the line equation `y = ax + b` can be derived, as well as anchor point `x0`.

#### a. Parameters
- `lane` (list of tuples):
    - list of `(x, y)` tuples representing 2D coords of lane points.

#### b. Returns
- tuple `(x0, a, b)`:
    - `x0` (float): anchor point, representing `(x0, y = 720)`.
    - `a` (float): slope.
    - `b` (float): y-intercept.

### 3. `getEgoIndexes()`

Identifies 2 ego lanes - left and right - from a sorted list of lane anchors.

Basically, left and right ego lanes are the 2 lanes closest to the center of the frame.
Leveraging those "anchor" points, I pick the 2 anchors closest to center point of bottom edge `(640, 720)`, left and right. Their lanes are ego lanes.

This is true like 99% of the time, and is a good heuristic for this dataset. Of course it might mess up if the car is not driving straight, but these datasets are mostly from a car cruising on highways, so it's fine ig.

#### a. Parameters

- `anchors` (list of tuples):
    - list of `(x, y)` tuples representing the anchors of lanes.
    - In those labels, lanes are labeled from left to right, so anchors extracted from them are also sorted x-coords in ascending order.

#### b. Returns
- `(left_ego_idx, right_ego_idx)` (tuple):
    - 2 indexes in the original lane list, indicating left and right ego lanes.

Sometimes there's no lanes on one side of the frame, so I return a string to indicate that.

### 4. `getDrivablePath()`

Computes drivable path as midpoint between 2 ego lanes, basically the main point of this task.

Average is taken with points having same y-coord. If not, skip to ensure alignment.

#### a. Parameters

- `left_ego` (list of tuples):
    - list of `(x, y)` points representing left ego lane.
- `right_ego` (list of tuples):
    - same as above, for right ego lane.

#### b. Returns
- `drivable_path` (list of tuples):
    - list of `(x, y)` points representing drivable path.

### 5. `annotateGT()`

Annotates and saves an image with:
- Raw image, in `output_dir/image`.
- Annotated image with all lanes, in `output_dir/visualization`.
- Binary segmentation mask of drivable path, in `output_dir/segmentation`.

#### a. Parameters

- `anno_entry` (dict):
    - an annotation entry containing:
        + `lanes` (list of list of tuples): a list of lane points, each represented as `(x, y)` tuples. Coords may be normalized (0 to 1) or absolute.
        + `ego_indexes` (list of int): indexes of ego lanes in the `lanes` list.
        + `drivable_path` (list of tuples): drivable path as a list of `(x, y)` tuples.
- `anno_raw_file` (str):
    - file path of raw input image to annotate.
- `raw_dir` (str):
    - directory to save raw (unlabeled) image copy.
- `visualization_dir` (str):
    - directory to save annotated (labeled) image.
- `mask_dir` (str):
    - directory to save binary segmentation mask.
- `normalized` (bool, optional):
    - defaults to `True`.
    - if `True`, all coords are scaled/normalized to `(0, 1)`. Otherwise, absolute.

#### b. Returns
No returns.

#### c. Notes
In visualization image, different lanes have different colors:
- Outer lanes: red.
- Ego lanes: green.
- Drivable path: yellow.

### 6. `parseAnnotations()`

Parses lane annotations from raw dataset file, then extracts normalized GT data.

First, read raw annotation/label data, then filter and process lane info, then identify 2 ego lanes, and calculate drivable path. All coords are normalized. Basically a "main" function.

#### a. Parameters

- `anno_path` (str):
    - path to annotation file containing lane data in JSON lines format.

#### b. Returns
- `anno_data` (dict):
    - dictionary mapping `raw_file` paths to their corresponding processed annotations.
    - each entry contains:
        + `lanes` (list of list of tuples): normalized lane points for each lane.
        + `ego_indexes` (tuple): indexes of 2 left and right ego lanes.
        + `drivable_path` (list of tuples): normalized points of the drivable path.
        + `img_width` (float): image width. TuSimple is 1280.
        + `img_height` (float): image height. TuSimple is 720.

#### c. Notes
- Lanes with fewer than 2 valid points `(x != 2)` are ignored.
- All coords are normalized, as requested by Mr. Zain.
- Warnings are issued for frames with no lanes on one side, while finding ego indexes.

## II. Workflow & usage

### 1. Workflow

1. Read raw annotation/label data, get all lane info.
2. Determine ego lanes:
    - Calculate lane “anchors”.
    - Determine 2 anchors of 2 ego lanes.
3. Determine drivable path.
4. Parse everything to new index, all coords normalized.
5. Save a copy of raw img, and a labeled img with ego, drivable path, & others.

### 2. Usage

#### a. Cmd line args

- `dataset_dir` : str
    - path to TuSimple dataset directory.
    - only accepts the dir right after extraction. So it should be `<smth>/tu_simple` if you tried to download it from Kaggle.
- `output_dir` : str
    - path to output directory where processed files will be stored. These dirs can either be relative or absolute.

#### b. Example

```
`python process_tusimple.py --dataset_dir /path/to/TuSimple --output_dir /path/to/output`
```

Structure of `output_dir`:
```
--output_dir
    |----image
    |----segmentation
    |----visualization
    |----drivable_path.json
```

python3 EgoPath/create_path/TuSimple/process_tusimple.py --dataset_dir ../pov_datasets/ --output_dir ../pov_datasets/TUSIMPLE
python3 EgoPath/create_path/TuSimple/parse_tusimple_bev.py --dataset_dir ../pov_datasets/TUSIMPLE/
