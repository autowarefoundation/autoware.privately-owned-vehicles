# Autoware Privately Owned Vehicles: Safety Sheild

This folder contains the code and data for the *SafetyShield* module.

---
## Python POC

## Prerequisites
waymo open dataset
tensorflow
filterpy


## Tree
├── dataset
│   ├── cipo
│   │   ├── cipo
│   │   │   └── training
│   ├── lane_annotation
│   └── test
├── output
│   └── test_output
└── scripts


The folder contains a dataset folder which should contain the OpenLane dataset cipo labels and the corrresponding datarecords (from Waymo). Note that the filenames of the records must match the cipo labels names as the original naming convention.
For example record 

```
SafetyShield/dataset/test/individual_files_validation_segment-1024360143612057520_3580_000_3600_000_with_camera_labels.tfrecord
```
The corresponding cipo labels folder should be
```
segment-1024360143612057520_3580_000_3600_000_with_camera_labels
```

The script folder containes the scripts for the pipeline

The output directory should be used to save/load the plot/matrices/saved images.

## Usage

To run the estimation code
```
cd SafetyShield
python3 main.py scripts/main.py path/to/output/vehicle_to_pixel_la.npy path/to/dataset/test/ path/to/dataset/cipo/cipo/training/ path/to/output/test_output/
```

This will output the timeseries plot of L2 error and for ego and object velocities. By passing the --visualize flag, the code will also save perframe annotated image. Note however that this is slow and could consume up a lot of space if run with a lot of input data.

The plot_error.py script calculates the L2 error plot which is saved as an npz file for a better analysis.

To compute the homography matrix using openlane lane3d annotations,
```
python3 scripts/compute_homography.py /path/to/lane_annotation/segment-10289507859301986274_4200_000_4220_000_with_camera_labels/155784739267258100.json h_mat.npy
```
