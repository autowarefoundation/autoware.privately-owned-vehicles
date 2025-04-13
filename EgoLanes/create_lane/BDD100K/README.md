# EgoLanes Dataset - BDD100K

BDD100K provides comprehensive lane marking annotations across 100,000 diverse driving images (1280×720). The lane annotations include:

- Detailed polygonal representations of lane boundaries
- Lane direction attributes (parallel, vertical)
- Lane type classifications (solid, dashed, double)
- Binary lane segmentation masks
- Annotations across diverse conditions (day/night, clear/rainy, city/highway)

Format of lane_train.json
```json
[
  {
    "name": "0000f77c-6257be58.jpg",
    "labels": [
      {
        "id": "0",
        "attributes": {
          "laneDirection": "parallel",
          "laneStyle": "solid",
          "laneTypes": "road curb"
        },
        "category": "road curb",
        "poly2d": [
          {
            "vertices": [[503.67, 373.13], [357.79, 374.67]],
            "types": "LL",
            "closed": false
          }
        ]
      },
      ...  // Additional lane markings
    ]
  },
  {
    "name": "0000f77c-62c2a288.jpg",
    "labels": [
      ...  // Multiple lane markings with different attributes
      {
        "id": "6",
        "attributes": {
          "laneDirection": "parallel",
          "laneStyle": "dashed",
          "laneTypes": "single white"
        },
        "category": "single white",
        "poly2d": [
          {
            "vertices": [
              [795.68, 345.23], 
              [628.16, 370.58], 
              [504.73, 482.99], 
              [370.28, 571.15]
            ],
            "types": "LCCC",  // Line followed by curves
            "closed": false
          }
        ]
      },
      ...  // More lane markings
    ]
  },
  ...  // Thousands more images with similar annotation structure
]
```
</div>

Download image dataset - [Ground Truth Images](https://dl.cv.ethz.ch/bdd100k/data/100k_images_train.zip)<br>
Download lane_train.json, binary lane masks - [Lane_labels](https://dl.cv.ethz.ch/bdd100k/data/bdd100k_lane_labels_trainval.zip)<br>
Download the dataset - [All links](https://dl.cv.ethz.ch/bdd100k/data/)<br>
More about the dataset - [Academic Paper](https://arxiv.org/abs/1805.04687)<br>

# `process_bdd100k.py`

## Overview
This script processes BDD100K's `lane_train.json` data to generate standardized lane annotations for autonomous driving. It automatically classifies lanes into egoleft_lane, egoright_lane, and other_lanes categories based on their position relative to the vehicle. Adjacent lanes with similar slopes and proximity are intelligently merged to create continuous lane markings. The tool also handles sparse points through interpolation, normalizes coordinates, and generates annotated images for visual verification. The output `bdd100k_egolanes_train.json` provides a consistent, ready-to-use format for training lane detection models.

## Requirements
- Python 3.6+
- OpenCV
- PIL (Pillow)
- NumPy

## Usage

```bash
python process_bdd100k.py --image_dir /path/to/bdd100k/images/100k/ \
                          --labels_dir /path/to/lane_labels \
                          --output_dir ./output
```

## Arguments
- `--image_dir`: BDD100K ground truth image parent directory (right after extraction)
- `--labels_dir`: BDD100K lane labels directory (contains lane polygons and masks)
- `--output_dir`: Desired output directory
- `--crop`: (optional)[TOP, RIGHT, BOTTOM, LEFT] cropping values (default: [0, 140, 220, 140])

## Directory Structure

```bash
├── README.md
├── process_bdd100k.py
├── output
│   ├── bdd100k_egolanes_train.json # Classified, merged lane keypoints data
│   ├── images # Groundtruth images renamed
│   │   ├── 000000.png
│   │   └── 000001.png
│   ├── Segmentation # Binary mask for lane lines
│   │   ├── 000000.png
│   │   └── 000001.png
│   ├── visualization # Annotated images output
│   │   ├── 000000.png
│   │   └── 000001.png
```

## Core functions

### `classify_lanes()`
The heart of the script, this function analyzes lane markings to determine which ones represent the ego-vehicle's lane boundaries. It:
- Calculates anchor points for each lane by projecting them to the bottom of the image
- Identifies lanes closest to the image center as ego-vehicle lanes
- Merges adjacent lanes with similar slopes using angle-based comparison
- Handles special cases like when lanes extend beyond image boundaries
- Organizes lanes into structured categories for downstream processing

### `getLaneAnchor()`
Determines the "anchor point" of each lane by:
- Sorting lane points by y-coordinate (from bottom to top of image)
- Calculating the slope and y-intercept from the lowest two points
- Extrapolating to find where the lane would intersect the bottom of the image
- Handling special cases for vertical and horizontal lanes
- Returning the x-coordinate of this intersection as the lane's anchor point
This anchor is crucial for determining lane positions relative to the vehicle and establishing lane order from left to right.

### `getEgoIndexes()`
Identifies which lanes represent the ego-vehicle's lane boundaries by:
- Examining the sorted list of lane anchors (from left to right)
- Finding the first lane with an anchor position right of image center
- Taking this lane and the one before it as the ego-right and ego-left lanes
- Handling edge cases (e.g., all lanes on one side of the image, single lane)
- Returning indices of the two lanes that bound the ego-vehicle's path
This function effectively determines which lanes are most relevant for the vehicle's immediate trajectory.

### `interpolated_list()`
Ensures lane continuity by creating evenly-spaced points along sparse lane markings. It:
- Takes minimal points and generates a specified number of interpolated points
- Sorts points by y-coordinate to handle both straight and curved lanes
- Uses linear interpolation between adjacent points to create smooth transitions
- Critical for maintaining lane integrity after cropping operations

### `format_data()`
Prepares the final output by normalizing and formatting lane data. It:
- Applies cropping parameters to focus on relevant image regions
- Normalizes coordinates to 0-1 range for consistent model training
- Automatically interpolates lanes with too few points
- Creates the standardized data structure used in the output JSON file
