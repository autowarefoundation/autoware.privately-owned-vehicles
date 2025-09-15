# VisionPilot ROS2 - Middleware Wrapper Layer

ROS2-specific implementation that wraps framework-agnostic core engines from `VisionPilot/common`. This layer handles ROS2 message passing, topic management, and node orchestration while delegating core AI processing to shared common engines.

## Architecture Overview

The ROS2 layer acts as a thin wrapper around common engines:

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│   SENSORS   │───▶│    MODELS    │───▶│ VISUALIZATION   │
│             │    │              │    │                 │
│ ROS2 Video  │    │ ROS2 Model   │    │ ROS2 Viz        │
│ Publishers  │    │ Wrappers     │    │ Wrappers        │
│     │       │    │     │        │    │     │           │
│             │    │              │    │                 │
│ Common      │    │ Common       │    │ Common          │
│ Video       │    │ Inference    │    │ Visualization   │
│ Engine      │    │ Backends     │    │ Engines         │
└─────────────┘    └──────────────┘    └─────────────────┘
```

## Package Structure

- **`sensors/`** - ROS2 video publishing nodes + common video engine
- **`models/`** - ROS2 inference nodes + common AI backends  
- **`visualization/`** - ROS2 visualization nodes + common rendering engines

## Core Features

- **Framework Agnostic Core**: All AI processing handled by `VisionPilot/common` engines
- **ROS2 Integration**: Native ROS2 topics, parameters, and launch files
- **Configurable**: YAML-driven configuration with no hardcoded paths
- **Performance Monitoring**: Built-in latency and FPS measurements
- **Concurrent Execution**: Multiple independent pipelines simultaneously

## Prerequisites

### System Requirements

- Ubuntu 20.04/22.04
- ROS2 Humble
- CUDA 11.8+ (for GPU inference)
- OpenCV 4.x

### Dependencies

```bash
# ROS2 packages
sudo apt install ros-humble-cv-bridge ros-humble-image-transport

# Build tools
sudo apt install cmake build-essential

# ONNX Runtime (Download from releases)
# TensorRT (NVIDIA SDK)
```

## 🔨 Build Instructions

### 1. Clone and Build

```bash
cd VisionPilot/ROS2
colcon build --symlink-install --packages-select sensors models visualization \
  --cmake-args \
  -DONNXRUNTIME_ROOTDIR=/path/to/onnxruntime \
  -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4 \
  -DCMAKE_BUILD_TYPE=Release
```

### 2. Source the Workspace

```bash
source install/setup.bash
```

## Usage Examples

### Basic Video Publishing

```bash
ros2 run sensors video_publisher_node_exe \
  --ros-args \
  -p video_path:=../data/video.mp4 \
  -p output_topic:=/sensors/video/image_raw \
  -p frame_rate:=30.0
```

### Individual Model Inference

#### Scene Segmentation

```bash
ros2 launch models auto_seg.launch.py model_name:=scene_seg_model
```

#### Domain Segmentation  

```bash
ros2 launch models auto_seg.launch.py model_name:=domain_seg_model
```

#### Depth Estimation

```bash
ros2 launch models auto_3d.launch.py
```

### Visualization

#### Scene Segmentation Visualization

```bash
ros2 launch visualization visualize_scene_seg.launch.py
```

#### Domain Segmentation Visualization

```bash
ros2 launch visualization visualize_domain_seg.launch.py
```

#### Depth Visualization

```bash
ros2 launch visualization visualize_depth.launch.py
```

### Complete Pipelines

#### Full Scene Segmentation Pipeline

```bash
ros2 launch models run_pipeline.launch.py \
  pipeline:=scene_seg \
  video_path:="../data/video.mp4"
```

#### Full Depth Pipeline  

```bash
ros2 launch models run_pipeline.launch.py \
  pipeline:=scene_3d \
  video_path:="../data/video.mp4"
```

## Configuration

### YAML Configuration Files

All model parameters are controlled via YAML files - **single source of truth**:

#### `models/config/autoseg.yaml`

```yaml
scene_seg_model:
  ros__parameters:
    model_path: "../data/models/SceneSeg_FP32.onnx"
    backend: "tensorrt"      # onnxruntime | tensorrt
    precision: "fp16"        # fp16 | fp32 | cpu | cuda
    model_type: "segmentation"
    input_topic: "/sensors/video/image_raw"
    output_topic: "/autoseg/scene_seg/mask"
    measure_latency: true    # Enable/disable performance monitoring

domain_seg_model:
  ros__parameters:
    model_path: "../data/models/DomainSeg_FP32.onnx"
    backend: "tensorrt"
    precision: "fp32"
    model_type: "segmentation"
    input_topic: "/sensors/video/image_raw"
    output_topic: "/autoseg/domain_seg/mask"
    measure_latency: true
```

#### `models/config/auto3d.yaml`

```yaml
scene3d_model:
  ros__parameters:
    model_path: "../data/models/Scene3D_FP32.onnx"
    backend: "tensorrt"
    precision: "fp16"
    model_type: "depth"
    input_topic: "/sensors/video/image_raw"
    output_topic: "/auto3d/scene_3d/depth_map"
    measure_latency: true
```

### Model Integration

The system uses a **single generic inference node** (`run_model_node`) that delegates processing to common backends. Model behavior is determined by configuration and handled transparently by the common layer.

### Visualization Configuration

#### Scene Segmentation Colors

- **Background**: Black (0,0,0)
- **Foreground**: Red (255,0,0)
- **Class IDs**: 0=background, 1=foreground, 2=background

#### Domain Segmentation Colors  

- **Road**: Orange (255,93,61)
- **Off-road**: Blue (28,148,255)
- **Class IDs**: 0=road, 255=off-road

## Performance Monitoring

### Latency Measurements

Both inference and visualization nodes provide real-time performance metrics:

```bash
# Example output:
[scene_seg_model]: Frame 700: Inference Latency: 11.95 ms (83.7 FPS)
[scene_seg_viz]: Frame 300: Visualization Latency: 25.07 ms (39.9 FPS)
```

### Performance Interpretation

- **Inference Latency**: Pure model execution time (preprocessing + inference + post-processing)
- **Visualization Latency**: Color mapping + blending time
- **FPS Calculation**: `1000ms / latency_ms` (theoretical maximum throughput)
- **Monitoring Interval**: Every 100 frames (configurable)

### Performance Tuning

```bash
# Disable latency monitoring for production
measure_latency: false

# Backend optimization
backend: "tensorrt"    # Fastest for NVIDIA GPUs
precision: "fp16"      # 2x speedup with minimal accuracy loss

# CPU fallback
backend: "onnxruntime"
precision: "cpu"       # Most compatible
```

## Topic Structure

### Standard Topics

```bash
/sensors/video/image_raw           # Input video feed

# Scene Segmentation
/autoseg/scene_seg/mask           # Raw segmentation mask
/autoseg/scene_seg/viz            # Colorized visualization

# Domain Segmentation  
/autoseg/domain_seg/mask          # Raw segmentation mask
/autoseg/domain_seg/viz           # Colorized visualization

# Depth Estimation
/auto3d/scene_3d/depth_map        # Raw depth map
/auto3d/scene_3d/depth_viz        # Colorized depth visualization
```

### Debugging Topics

```bash
# Monitor topic rates
ros2 topic hz /autoseg/scene_seg/mask

# View raw messages
ros2 topic echo /autoseg/scene_seg/mask --max-count=1

# List all active topics
ros2 topic list | grep -E "(autoseg|auto3d|sensors)"
```

## Troubleshooting

### Common Issues

#### CUDA/TensorRT Issues

```bash
# Check CUDA installation
nvidia-smi

# Verify library paths
echo $LD_LIBRARY_PATH

# Use CPU fallback
precision: "cpu"
backend: "onnxruntime"
```

#### Model Loading Errors

```bash
# Verify model file exists
ls -la ../data/models/

# Check YAML configuration
cat models/config/autoseg.yaml

# Verify file permissions
chmod 644 ../data/models/*.onnx
```

#### Performance Issues

```bash
# Monitor system resources  
htop

# Check ROS node CPU usage
ros2 node info /scene_seg_model

# Reduce visualization quality
# (lower resolution, simpler color maps)
```

## Advanced Usage

### Running Multiple Models Simultaneously

```bash
# Terminal 1: Video publisher
ros2 run sensors video_publisher_node_exe \
  --ros-args -p video_path:=../data/video.mp4

# Terminal 2: Scene segmentation
ros2 launch models auto_seg.launch.py model_name:=scene_seg_model

# Terminal 3: Domain segmentation  
ros2 launch models auto_seg.launch.py model_name:=domain_seg_model

# Terminal 4: Scene visualization
ros2 launch visualization visualize_scene_seg.launch.py

# Terminal 5: Domain visualization
ros2 launch visualization visualize_domain_seg.launch.py
```

### Custom Model Integration

1. **Add model to YAML**:

   ```yaml
   your_model:
     ros__parameters:
       model_path: "../data/models/YourModel.onnx"
       model_type: "segmentation"  # or "depth" 
       # ... other parameters
   ```

2. **Launch with custom model**:

   ```bash
   ros2 launch models auto_seg.launch.py model_name:=your_model
   ```
