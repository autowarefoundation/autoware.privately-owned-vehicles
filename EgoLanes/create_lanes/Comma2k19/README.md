## process_comma2k19.py
## comma2k19 dataset
[comma.ai](https://comma.ai) presents comma2k19, a dataset of over 33 hours of commute in California's 280 highway. This means 2019 segments, 1 minute long each, on a 20km section of highway driving between California's San Jose and San Francisco. comma2k19 is a fully reproducible and scalable dataset. The data was collected using comma [EONs](https://comma.ai/shop/products/eon-gold-dashcam-devkit/) that has sensors similar to those of any modern smartphone including a road-facing camera, phone GPS, thermometers and 9-axis IMU. Additionally, the EON captures raw GNSS measurements and all CAN data sent by the car with a comma [grey panda](https://comma.ai/shop/products/panda-obd-ii-dongle/). 


## Downloads
Original link: The total dataset is ~100GB and can be downloaded [here](http://academictorrents.com/details/65a2fbc964078aff62076ff4e103f18b951c5ddb) It is divided into ~10GB chunks.
Note: Original link needs torrent to download

Alternative link: https://huggingface.co/datasets/commaai/comma2k19/tree/main


## Dataset Structure

#### Directory Structure
The data is split into 10 chunks of each about 200 minutes of driving. Chunks 1-2 of the dataset are of the RAV4 and the rest is the civic. The dongle_id of the RAV4 is `b0c9d2329ad1606b` and that of the civic is `99c94dc769b5d96e`.
```
Dataset_chunk_n
|
+-- route_id (dongle_id|start_time)
    |
    +-- segment_number
        |
        +-- preview.png (first frame video)
        +-- raw_log.bz2 (raw capnp log, can be read with openpilot-tools: logreader)
        +-- video.hevc (video file, can be read with openpilot-tools: framereader)
        +-- processed_log/ (processed logs as numpy arrays, see format for details)
        +-- global_pos/ (global poses of camera as numpy arrays, see format for details)
```

#### Pose Format
```
The poses of the camera and timestamps of every frame of the video are stored
as follows:
  frame_times: timestamps of video frames in boot time (s)
  frame_gps_times: timestamps of video frames in gps_time: ([gps week (weeks), time-of-week (s)])
  frame_positions: global positions in ECEF of camera(m)
  frame_velocities: global velocity in ECEF of camera (m/s)
  frame_orientations: global orientations as quaternion needed to
                      rotate from ECEF  frame to local camera frame
                      defined as [forward, right, down] (hamilton quaternion!!!!)
```
## Usage

#### a. Cmd line args

- `dataset_root_path` : str
    - path to Comma2k19 dataset directory (Where Chunk folders are present).
- `out_path` : str
    - Path to the output directory.
- `df` : int
    - Factor by which to downsample the frames.

#### b. Example

```
`python process_comma2k19.py --dataset_root_path /path/to/Comma2k19 --out_path /path/to/output --df 10`
```

Structure of outputs in each segment`:
```
Dataset_chunk_n
|
+-- route_id (dongle_id|start_time)
    |
    +-- segment_number
        |----image
        |----segmentation_masks
        |----visualization
        |----drivable_path.json
```
