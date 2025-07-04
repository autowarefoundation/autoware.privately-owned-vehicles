The data set is acquired from: https://github.com/once-3dlanes/once_3dlanes_benchmark

The scripts converts lane marking points from 3D coordinates in world metric frame to image pixel frame, saves the annotated image and yaml files to be used by PathFinder in the test/ folder.
```sh
cd autoware.privately-owned-vehicles/PathFinder/scripts/
python3 calc_img_to_world.py 
``` 

The generated images and yaml files are located here
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