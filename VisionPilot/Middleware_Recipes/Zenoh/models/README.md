# Models

This project demonstrates using Zenoh to run various models.

- `video_visualization` (`video_visualization.cpp`): Processes an input video file and saves a new video with the segmentation results overlaid.

## Usage

Switch the install folder

### Video Visualization

Subscribe a video from a Zenoh publisher and then publish it to a Zenoh Subscriber.

- Run the video publisher

```bash
./video_publisher -k video/input ../../data/video.mp4
```

- Run the model you want

```bash
# SceneSeg
./run_model ../../data/models/SceneSeg_FP32.onnx -i video/input -o video/output -m "segmentation"
# DomainSeg
./run_model ../../data/models/DomainSeg_FP32.onnx -i video/input -o video/output -m "segmentation"
# Scene3D
./run_model ../../data/models/Scene3D_FP32.onnx -i video/input -o video/output -m "depth"
```

- Subscribe the video

```bash
# Only the output
./video_subscriber -k video/output
# Combine the output and the raw video
./segment_subscriber -k video/input -s video/output
```
