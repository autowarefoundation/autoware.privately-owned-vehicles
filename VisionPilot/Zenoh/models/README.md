# Models

This project demonstrates using Zenoh to run various models.

- `video_visualization` (`video_visualization.cpp`): Processes an input video file and saves a new video with the segmentation results overlaid.

## Build

- Environment setup

```shell
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

- Configure with cmake

```shell
mkdir build && cd build
cmake .. \
    -DLIBTORCH_INSTALL_ROOT=/path/to/libtorch/ \
    -DONNXRUNTIME_ROOTDIR=/path/to/onnxruntime-linux-x64-gpu-1.22.0 \
    -DUSE_CUDA_BACKEND=True
```

- Build

```shell
make
```

## Usage

After a successful build, you will find two executables in the `build` directory.

### Video Visualization

Subscribe a video from a Zenoh publisher and then publish it to a Zenoh Subscriber.

- Run the video publisher

```bash
./video_publisher -k video/input
```

- Run the model you want

```bash
# SceneSeg
./run_model SceneSeg_FP32.onnx -i video/input -o video/output -m "segmentation"
# DomainSeg
./run_model DomainSeg_FP32.onnx -i video/input -o video/output -m "segmentation"
# Scene3D
./run_model Scene3D_FP32.onnx -i video/input -o video/output -m "depth"
```

- Subscribe the video

```bash
# Only the output
./video_subscriber -k video/output
# Combine the output and the raw video
./segment_subscriber -k video/input -s video/output
```
