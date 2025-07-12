## ONCE_3DLanes Data Parsing Script
The data set is acquired from: https://github.com/once-3dlanes/once_3dlanes_benchmark

The script converts lane marking points from 3D coordinates in world metric frame to image pixel frame, saves the annotated image, as well as the lane marking and camera information in yaml files to be used by PathFinder in the test/ folder.
```sh
cd autoware.privately-owned-vehicles/PathFinder/scripts/
python3 calc_img_to_world.py 
``` 

The generated annotated images and yaml files are located here
```
├── PathFinder
|   |   ...
│   ├── ONCE_3DLanes
│   │   ├── data
│   │   ├── list
│   │   ├── train
│   │   └── val
│   ├── README.md
│   ├── scripts
│   │   ├── calc_img_to_world.py
│   │   └── README.md
│   ├── src
│   │   └── path_finder.cpp
│   └── test
│       ├── 000001 <---- new output
│       ├── image_to_world_transform.yaml
│       └── README.md
├── README.md

```
Example of annotated image
![](../docs/camera_view_lane_annotations.png)

YAML file structure
```yaml
camera_intri: # 4x4 camera intrinsics parameters
- - 958.3320922851562
  - -0.1807
  - 934.5001074545908
  - 0.0
- - 0.0
  - 961.3646850585938
  - 518.6117222564244
  - 0.0
- - 0.0
  - 0.0
  - 1.0
  - 0.0
lanes2d: # N number of lanes, 
         # which each consists of M number of 2D pixel coordinates
- - - 645.3815008913232
    - 549.904956390994
  - - 626.7568478169966
    - 555.1549561815966
  - - 609.1406738698744
    - 560.4573983539368
  - - 588.0431731769183
    - 565.7066064900592
  - - 567.1588414151392
    - 571.0207276363733
...
lanes3d: # N number of lanes,
         # which each consists of M number of 3D points in meters
- - - -14.625
    - 1.578
    - 48.478
  - - -13.677
    - 1.619
    - 42.592
  - - -12.924
    - 1.657
...

```