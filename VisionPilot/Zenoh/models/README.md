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

- Usage the video publisher and subscriber

```bash
# Terminal 1
./video_publisher -k video/input
# Terminal 2
./run_model SceneSeg_FP32.onnx -i video/input -o video/output
./run_model DomainSeg_FP32.onnx -i video/input -o video/output
# Terminal 3
./video_subscriber -k video/output
```
