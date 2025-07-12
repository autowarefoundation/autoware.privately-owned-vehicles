## Path Finder

C++ module responsible for fusing the outputs of EgoPath and EgoLanes and deriving the shape of the driving corridor and the relative position of the vehicle with respect to the driving corridor. It takes in labelled EgoLane and EgoPath detection in pixel coordinates from camera perspective and tracks error metrics (cross-track error, yaw error, curvature and corridor width) in metric coordinates from Bird's-Eye-View perspective using Bayes filter. These error metrics will be utilized by a downstream steering controller.

![](docs/bev.png)![](docs/camera_view_lane_annotations.png) 

## Dependencies
- C++17
- Eigen3
- OpenCV2

## Dataset
The data set is acquired from: https://github.com/once-3dlanes/once_3dlanes_benchmark

## How it works
1. Projects image pixels in camera perspective to BEV pixels
2. Curve fitting using quadratic polynomial
3. Calculate raw value for error metrics
4. Track error metrics across frames using Bayes filter

### Filter Implementation
The following measures are taken to ensure robust estimation of
1. Cross-Track Error: The width of the driving corridor is tracked by the filter so that when EgoPath is not available, CTE can be derived from immediate left/right EgoLanes CTE through offsetting by half of the width. A fused CTE is also available through the Gaussian product of individually tracked CTEs (i.e. EgoPath, left EgoLane, right EgoLane).
2. Yaw error & Curvature: Fused left and right EgoLanes Yaw error & Curvature to reduce the effect of perspective warping from the flat ground plane assumption. 

13 Tracked states including 10 individually tracked states and 3 fused states:
![alt text](docs/tracked_states.png)

## How to run

1. Download images and lane marking ground truth data separately from https://once-3dlanes.github.io/3dlanes/. Download the lane marking data from https://drive.google.com/file/d/16_Tw0K55yR-3sJf8-lBbymGHcq0H7Zjx/view and move the unzipped folder under `.../PathFinder/`. Download `raw_cam01_p0.tar` from https://drive.google.com/drive/folders/1gxxkM-K7lA2unT5cVQnZ1UxSOOshatZQ as we only need the forward facing images, move the `data` subfolder containing all the images into the previously created folder at `.../PathFinder/ONCE_3DLanes/`. The current state of the directory should look like
   ```
   PathFinder
   ├── ...
   ├── ONCE_3DLanes
   │   ├── data
   │   ├── list
   │   ├── train
   │   └── val
   ├── ...
   ``` 
2. Create folder to store yaml files and run python script
   ```sh 
   mkdir -p .../PathFinder/test/000001/img # run 000001
   cd .../PathFinder/scripts
   python3 calc_img_to_world.py
3. Build and run
    ```sh
    # inside .../PathFinder/
    mkdir build && cd build
    cmake ..
    make

    # Run
    ./PathFinder 
    # Press alt for next frame
    ```
## Next Steps
Full pipeline integration with upstream perception and downstream controller for testing in CARLA simulator. CARLA 0.10.0 with lower VRAM requirements: https://gist.github.com/xmfcx/a5e32fdecfcd85c6cc9d472ce7a3a98d