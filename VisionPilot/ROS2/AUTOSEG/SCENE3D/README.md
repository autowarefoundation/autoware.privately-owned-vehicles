# autoware_pov_scene3d

This ROS2 package performs real-time monocular depth estimation using optimized inference backends. It supports both ONNX Runtime and TensorRT backends with various precision modes for maximum performance flexibility.

## Features

- **Dual Backend Support**: Choose between ONNX Runtime and TensorRT
- **Multiple Precision Modes**: FP32, FP16 for neural network inference
- **Engine Caching**: TensorRT engines are automatically cached for faster startup
- **Real-time Performance**: Optimized for high-FPS depth estimation with latency monitoring
- **Depth Visualization**: Viridis colormap for beautiful depth visualization
- **Flexible Input**: Works with live cameras or video files

## Architecture

The package contains two main components:

1. **Scene3DNode**: A C++ composable node with pluggable inference backends
2. **video_publisher.py**: A Python script for testing with video files

### Inference Backends

#### ONNX Runtime Backend
- **Best for**: Rapid prototyping, cross-platform compatibility, CPU fallback
- **Precision options**: 
  - `"cpu"`: CPU inference
  - `"cuda"`: GPU inference with FP32 precision
- **Supports**: Standard ONNX models from PyTorch, TensorFlow

#### TensorRT Backend  
- **Best for**: Maximum performance on NVIDIA GPUs
- **Precision options**:
  - `"fp32"`: Full precision (baseline accuracy)
  - `"fp16"`: Half precision (~2x faster, minimal accuracy loss)
- **Engine Caching**: Automatically creates and reuses optimized engines
- **File naming**: `model.onnx.{precision}.engine`

## Topics

### Input
- `~/in/image` (sensor_msgs/Image): RGB input images

### Outputs
- `~/out/depth` (sensor_msgs/Image): Raw depth map (32-bit float, TYPE_32FC1)
- `~/out/color_depth` (sensor_msgs/Image): Colorized depth visualization (BGR8)

## Quick Start

### 1. Build the Package

```bash
# Build with ONNX Runtime and TensorRT support
colcon build --packages-select autoware_pov_scene3d \
  --cmake-args \
  -DONNXRUNTIME_ROOTDIR=/path/to/your/onnxruntime \
  -DCMAKE_BUILD_TYPE=Release
```

### 2. Test with Video File

```bash
# Source your workspace
source install/setup.bash

# Set library paths for both ONNX Runtime and TensorRT
export LD_LIBRARY_PATH=/path/to/onnxruntime/lib:/path/to/tensorrt/lib:$LD_LIBRARY_PATH

# Run test pipeline
ros2 launch autoware_pov_scene3d test_pipeline.launch.py \
  model_path:=data/scene3d_model.onnx \
  video_path:=data/test_video.mp4 \
  backend:=tensorrt \
  precision:=fp16 \
  measure_latency:=true
```

## Usage Examples

### TensorRT with FP16 (Recommended for Performance)

```bash
ros2 launch autoware_pov_scene3d test_pipeline.launch.py \
  model_path:=data/Scene3D_FP32.onnx \
  video_path:=data/driving_video.mp4 \
  backend:=tensorrt \
  precision:=fp16 \
  frame_rate:=30.0 \
  measure_latency:=true
```

**What happens:**
1. First run: Builds FP16 engine from FP32 ONNX model (takes 1-2 minutes)
2. Saves engine as `data/Scene3D_FP32.onnx.fp16.engine`
3. Subsequent runs: Loads pre-built engine instantly
4. Expected performance: ~2x faster than FP32

### TensorRT with FP32 (Maximum Accuracy)

```bash
ros2 launch autoware_pov_scene3d test_pipeline.launch.py \
  model_path:=data/Scene3D_FP32.onnx \
  backend:=tensorrt \
  precision:=fp32 \
  measure_latency:=true
```

**Engine file:** `data/Scene3D_FP32.onnx.fp32.engine`

### ONNX Runtime GPU

```bash
ros2 launch autoware_pov_scene3d test_pipeline.launch.py \
  model_path:=data/Scene3D_FP32.onnx \
  backend:=onnxruntime \
  precision:=cuda \
  measure_latency:=true
```

### ONNX Runtime CPU Fallback

```bash
ros2 launch autoware_pov_scene3d test_pipeline.launch.py \
  model_path:=data/Scene3D_FP32.onnx \
  backend:=onnxruntime \
  precision:=cpu \
  frame_rate:=10.0 \
  measure_latency:=false
```

## Live Camera Usage

For real-world deployment with a live camera:

```bash
ros2 launch autoware_pov_scene3d scene3d.launch.py \
  model_path:=/path/to/your/depth_model.onnx \
  backend:=tensorrt \
  precision:=fp16 \
  input_image_topic:=/sensing/camera/front/image_raw \
  measure_latency:=true
```

## Launch Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_path` | - | Path to ONNX depth estimation model |
| `backend` | `"onnxruntime"` | `"onnxruntime"` or `"tensorrt"` |
| `precision` | `"cpu"` | See precision options below |
| `gpu_id` | `0` | GPU device ID for inference |
| `measure_latency` | `false` | Enable latency monitoring (samples every 200 frames) |
| `input_image_topic` | `"/image_raw"` | Input camera topic |
| `frame_rate` | `30.0` | Video publisher frame rate (test only) |

### Precision Options by Backend

**ONNX Runtime:**
- `"cpu"`: CPU inference
- `"cuda"`: GPU inference (FP32)

**TensorRT:**
- `"fp32"`: Full precision GPU inference
- `"fp16"`: Half precision GPU inference

## Model Requirements

- **Input**: RGB images (any resolution, automatically resized to 640x320)
- **Output**: Single-channel depth map [batch, height, width] or [batch, 1, height, width]
- **Preprocessing**: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Supported formats**: ONNX models exported from PyTorch, TensorFlow, etc.

## Performance Tips

### Backend Selection Guide

**Use TensorRT when:**
- You need maximum performance
- Running FP32 or FP16 models
- Deploying on NVIDIA GPUs
- Inference speed is critical

**Use ONNX Runtime when:**
- Need cross-platform compatibility
- Rapid prototyping and testing
- CPU inference required
- Model format compatibility issues

### Engine Caching (TensorRT)

TensorRT automatically optimizes models for your specific hardware:

```
# First run with fp16
[tensorrt_backend]: No pre-built fp16 engine found. Building from ONNX model: data/model.onnx
[tensorrt_backend]: Building TensorRT engine with FP16 precision
[tensorrt_backend]: Saving fp16 engine to data/model.onnx.fp16.engine

# Subsequent runs with fp16  
[tensorrt_backend]: Found pre-built fp16 engine at data/model.onnx.fp16.engine
[tensorrt_backend]: Using TensorRT backend
```

Each precision gets its own engine file:
- `model.onnx.fp32.engine`
- `model.onnx.fp16.engine`

### Latency Monitoring

When `measure_latency:=true`, the node samples inference time every 200 frames:

```
Frame 200: Inference Latency: 8.42 ms (118.8 FPS)
Frame 400: Inference Latency: 7.91 ms (126.4 FPS)
```

## Visualization

View the depth estimation output:

```bash
# List available topics
ros2 topic list

# View colorized depth map
rqt_image_view ~/out/color_depth

# View raw depth values (grayscale)
rqt_image_view ~/out/depth
```

## Depth Processing

### Raw Depth Output
- **Format**: 32-bit float (TYPE_32FC1)
- **Values**: Model-dependent depth values
- **Use case**: Further processing, point cloud generation, SLAM

### Colorized Depth Output
- **Format**: BGR8 color image
- **Colormap**: Viridis (dark blue = close, bright yellow = far)
- **Normalization**: Per-frame min-max scaling
- **Use case**: Visualization, debugging, demonstrations

## Troubleshooting

### Common Issues

**"Failed to load library":**
```bash
# Ensure library paths are set correctly
export LD_LIBRARY_PATH=/path/to/onnxruntime/lib:/path/to/tensorrt/lib:$LD_LIBRARY_PATH
```

**"No optimization profile defined":**
- This happens with dynamic ONNX models. The TensorRT backend automatically handles this.

**"Unexpected output dimensions":**
- Ensure your model outputs single-channel depth maps
- Supported shapes: [N,H,W] or [N,1,H,W]

### Performance Benchmarking

Compare different configurations:

```bash
# Benchmark FP32
ros2 launch autoware_pov_scene3d test_pipeline.launch.py \
  backend:=tensorrt precision:=fp32 measure_latency:=true

# Benchmark FP16  
ros2 launch autoware_pov_scene3d test_pipeline.launch.py \
  backend:=tensorrt precision:=fp16 measure_latency:=true

# Benchmark ONNX Runtime
ros2 launch autoware_pov_scene3d test_pipeline.launch.py \
  backend:=onnxruntime precision:=cuda measure_latency:=true
```

## Integration Example

```bash
# Terminal 1: Launch depth estimation
ros2 launch autoware_pov_scene3d scene3d.launch.py \
  model_path:=models/midas_v2.onnx \
  backend:=tensorrt \
  precision:=fp16 \
  input_image_topic:=/camera/image_raw

# Terminal 2: Visualize results  
rqt_image_view ~/out/color_depth

# Terminal 3: Process depth for SLAM/navigation
ros2 run your_slam_package depth_processor \
  --ros-args -r depth:=~/out/depth
``` 