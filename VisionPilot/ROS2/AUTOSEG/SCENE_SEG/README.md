# autoware_pov_scene_seg

This ROS2 package performs semantic segmentation on camera images using an ONNX model. It is designed to be a modular and efficient component for a perception pipeline.

The core of the package is a C++ composable node that uses ONNX Runtime for hardware-accelerated inference.

## How It Works

The package contains two main components:

1.  `SceneSegNode`: A C++ composable node that performs the actual segmentation. It loads an ONNX model, subscribes to an image topic, runs inference, and publishes the resulting segmentation masks.
2.  `video_publisher.py`: A Python script used for testing. It reads a video file, and publishes its frames as `sensor_msgs/msg/Image` messages, simulating a live camera feed.

## Quickstart: Running the Test Pipeline

The easiest way to see the node in action is to use the provided test pipeline, which plays a video file and runs the segmentation on it.

### 1. Build the Package

First, build the workspace, making sure to point `cmake` to the location of your ONNX Runtime installation.

```bash
colcon build --packages-select autoware_pov_scene_seg --cmake-args -DONNXRUNTIME_ROOTDIR=/path/to/your/onnxruntime
```

### 2. Run the Test Launch File

The `test_pipeline.launch.py` file starts both the video publisher and the segmentation node and correctly connects them.

**IMPORTANT**: You must prepend `LD_LIBRARY_PATH` to the command so the ROS2 node can find the ONNX Runtime shared libraries (`.so` files) at runtime.

```bash
# Source your workspace
source install/setup.bash

# Set Library Path and Launch
LD_LIBRARY_PATH=/path/to/your/onnxruntime/lib:$LD_LIBRARY_PATH ros2 launch autoware_pov_scene_seg test_pipeline.launch.py \
    model_path:=/path/to/your/model.onnx \
    video_path:=/path/to/your/video.mp4 \
    precision:=cuda
```

**Launch Arguments:**

*   `model_path`: Absolute path to the ONNX model file.
*   `video_path`: Absolute path to the video file for the publisher.
*   `precision`: The execution provider to use. Can be `cpu` or `cuda`.
*   `gpu_id`: The integer ID of the GPU to use (e.g., `0`).

### 3. Visualize the Output

In a separate terminal, you can use `rqt_image_view` to see the colorized segmentation mask.

```bash
# List available topics
ros2 topic list

# View the output. The topic name will be something like /out/color_mask
rqt_image_view /out/color_mask
```

## Usage with a Live Camera

For a real-world application, you will use your own camera driver node instead of the video publisher. You will launch the `scene_seg.launch.py` file directly and remap the input topic to your camera's topic.

### 1. Identify Your Camera Topic

First, run your camera driver and find the name of the topic that publishes images. You can do this with:

```bash
ros2 topic list
```
Let's assume your camera publishes to `/sensing/camera/front/image_raw`.

### 2. Launch the Segmentation Node

Now, launch the `scene_seg.launch.py` file directly. Use the `input_image_topic` launch argument to tell the node which topic to subscribe to.

```bash
# Source your workspace
source install/setup.bash

# Set Library Path and Launch
LD_LIBRARY_PATH=/path/to/your/onnxruntime/lib:$LD_LIBRARY_PATH ros2 launch autoware_pov_scene_seg scene_seg.launch.py \
    model_path:=/path/to/your/model.onnx \
    input_image_topic:=/sensing/camera/front/image_raw \
    precision:=cuda
```

This will start only the segmentation node, which will listen to your camera, process the images, and publish the results. You can visualize the output with `rqt_image_view` as before.
