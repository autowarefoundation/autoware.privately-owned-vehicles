## MUSES

#### Academic paper: https://arxiv.org/abs/2401.12761

[MUSES, the MUlti-SEnsor Semantic perception dataset](https://muses.vision.ee.ethz.ch/), designed for driving under increased uncertainty and adverse conditions. MUSES comprises a diverse collection of images, evenly distributed across different combinations of weather conditions (clear, fog, rain, and snow) and illuminations (day time/night time). The dataset contains camera images with a 3D LIDAR pointcloud from a MEMS LIDAR scanner (depth up to 200m). A depth-map can be calculated by projecting the 3D LIDAR points on to the camera image frame. After temporal downsampling, the dataset yields **a total of 2,500 training samples with associated ground truth data.**

### process_muses.py

**Arguments**
```bash
python3 process_muses.py -r <data directory> -s <save directory>

-r, --root : root data directory filepath
-s, --save : root save directory filepath
```
**Dataset Parameters**
- Camera height above road surface: 1.4m
- Max Height in height map: 7m
- Min Height in height map: -5m