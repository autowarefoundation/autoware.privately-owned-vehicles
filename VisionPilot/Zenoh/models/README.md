# Models

This project demonstrates using Zenoh to run various models.

* `video_visualization` (`video_visualization.cpp`): Processes an input video file and saves a new video with the segmentation results overlaid.

## Dependencies

* **CUDA**: Optional for GPU processing.
* **OpenCV**: For image and video processing.
  * Ubuntu: `sudo apt install libopencv-dev`
* **ONNX Runtime**: For model inference.
  * Download from [the GitHub release](https://github.com/microsoft/onnxruntime/releases)
* **LibTorch**: For tensor manipulation capabilities.
  * Download from [the PyTorch website](https://pytorch.org/get-started/locally/)
* **Zenoh C library**: Required for the transportation.
  * Download from [the GitHub release](https://github.com/eclipse-zenoh/zenoh-c/releases)
  * You can also add the Eclipse repository for apt server.
  
    ```shell
    echo "deb [trusted=yes] https://download.eclipse.org/zenoh/debian-repo/ /" | sudo tee -a /etc/apt/sources.list > /dev/null
    sudo apt update
    sudo apt install libzenohc-dev
    ```

* **CLI11**: Used for the command line interface.
  * Ubuntu: `sudo apt install libcli11-dev`

* **SceneSeg Model and Weights**
  * [Link to Download Pytorch Model Weights *.pth](https://drive.google.com/file/d/1vCZMdtd8ZbSyHn1LCZrbNKMK7PQvJHxj/view?usp=sharing)
  * [Link to Download Traced Pytorch Model *.pt](https://drive.google.com/file/d/1G2pKrjEGLGY1ouQdNPh11N-5LlmDI7ES/view?usp=drive_link)
  * [Link to Download ONNX FP32 Weights *.onnx](https://drive.google.com/file/d/1l-dniunvYyFKvLD7k16Png3AsVTuMl9f/view?usp=drive_link)

## Build

* Environment setup

```shell
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

* Configure with cmake

```shell
mkdir build && cd build
cmake .. \
    -DLIBTORCH_INSTALL_ROOT=/path/to/libtorch/ \
    -DONNXRUNTIME_ROOTDIR=/path/to/onnxruntime-linux-x64-gpu-1.22.0 \
    -DUSE_CUDA_BACKEND=True
```

* Build

```shell
make
```

## Usage

After a successful build, you will find two executables in the `build` directory.

### Video Visualization

Subscribe a video from a Zenoh publisher and then publish it to a Zenoh Subscriber.

* Usage the video publisher and subscriber

```bash
# Terminal 1
./video_publisher -k video/input
# Terminal 2
./video_visualization SceneSeg_FP32.onnx -i video/input -o video/output
./video_visualization DomainSeg_FP32.onnx -i video/input -o video/output
# Terminal 3
./video_subscriber -k video/output
```
