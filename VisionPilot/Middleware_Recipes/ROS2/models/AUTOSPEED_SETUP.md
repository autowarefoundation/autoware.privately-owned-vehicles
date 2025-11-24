# AutoSpeed Implementation Summary

## What Was Created

### 1. Backend Implementation
**Location**: `VisionPilot/common/backends/autospeed/`

- `tensorrt_engine.hpp` - AutoSpeed-specific TensorRT backend header
- `tensorrt_engine.cpp` - Backend implementation with:
  - PyTorch → ONNX conversion support
  - Letterbox preprocessing (640x640 with aspect ratio preservation)
  - Simple normalization (÷255, no ImageNet normalization)
  - TensorRT engine building and caching
  - Transformation parameter tracking

### 2. ROS2 Node Implementation
**Location**: `VisionPilot/ROS2/models/`

- `include/run_autospeed_node.hpp` - Node header
- `src/run_autospeed_node.cpp` - Node implementation with:
  - Post-processing pipeline (confidence filtering + NMS)
  - Coordinate transformation (letterbox → original image)
  - Detection publishing (`vision_msgs::Detection2DArray`)

### 3. Documentation
- `common/backends/autospeed/README.md` - Complete technical documentation

## Key Architecture Decisions

### ✅ Separated from Existing Backend
- Created new `autospeed/` folder - does NOT modify existing `tensorrt_backend.cpp`
- Clean separation allows different preprocessing strategies

### ✅ Flexible Input Format
- Accepts **PyTorch checkpoints** (`.pt`) OR **ONNX models** (`.onnx`)
- Automatic conversion pipeline: `.pt` → `.onnx` → `.engine`
- Engines cached for fast subsequent loads

### ✅ AutoSpeed-Specific Preprocessing
```cpp
// Standard backend (SceneSeg/Scene3D)
normalize_imagenet(image)  // (x - mean) / std

// AutoSpeed backend
letterbox_resize(image)    // Maintain aspect ratio
normalize_simple(image)    // x / 255.0
```

### ✅ Complete Post-Processing
- Confidence filtering
- NMS (Non-Maximum Suppression)
- Coordinate transformation
- Bounding box clamping

## What You Need to Do Next

### Step 1: Update CMakeLists.txt

You'll need to add the new files to your build system. Here's what to add to `VisionPilot/ROS2/models/CMakeLists.txt`:

```cmake
# Add AutoSpeed backend source
set(AUTOSPEED_BACKEND_SOURCES
  ../../common/backends/autospeed/tensorrt_engine.cpp
)

# Create AutoSpeed backend library
add_library(autospeed_backend SHARED
  ${AUTOSPEED_BACKEND_SOURCES}
)

target_include_directories(autospeed_backend PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../common/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../common/backends/autospeed
)

target_link_libraries(autospeed_backend
  ${OpenCV_LIBS}
  nvinfer
  nvonnxparser
  ${CUDA_LIBRARIES}
)

# Create AutoSpeed node library
add_library(run_autospeed_node SHARED
  src/run_autospeed_node.cpp
)

target_link_libraries(run_autospeed_node
  autospeed_backend
  ${OpenCV_LIBS}
  ${rclcpp_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
  ${vision_msgs_LIBRARIES}
  image_transport::image_transport
)

rclcpp_components_register_node(run_autospeed_node
  PLUGIN "autoware_pov::vision::RunAutoSpeedNode"
  EXECUTABLE autospeed_node_exe
)

# Install AutoSpeed node
install(TARGETS
  autospeed_backend
  run_autospeed_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
```

### Step 2: Create Launch File

Create `VisionPilot/ROS2/models/launch/autospeed.launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    return LaunchDescription([
        ComposableNodeContainer(
            name='autospeed_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[
                ComposableNode(
                    package='models',
                    plugin='autoware_pov::vision::RunAutoSpeedNode',
                    name='autospeed_node',
                    parameters=[{
                        'model_path': '/path/to/your/autospeed/best.pt',  # CHANGE THIS
                        'precision': 'fp16',
                        'gpu_id': 0,
                        'conf_threshold': 0.6,
                        'iou_threshold': 0.45,
                        'input_topic': '/sensors/camera/image_raw',
                        'output_topic': '/autospeed/detections'
                    }],
                    extra_arguments=[{'use_intra_process_comms': True}]
                )
            ],
            output='screen'
        )
    ])
```

### Step 3: Update package.xml

Add dependency for `vision_msgs` in `VisionPilot/ROS2/models/package.xml`:

```xml
<depend>vision_msgs</depend>
```

### Step 4: Build and Test

```bash
# Navigate to ROS2 workspace
cd VisionPilot/ROS2

# Build
colcon build --packages-select models

# Source
source install/setup.bash

# Launch
ros2 launch models autospeed.launch.py
```

## How It Works - Complete Flow

### 1. Initialization
```
User provides: best.pt + precision=fp16
    ↓
Backend checks: Does best.onnx exist?
    ↓ No
Convert: best.pt → best.onnx (via Python/torch.onnx.export)
    ↓
Backend checks: Does best.onnx.fp16.engine exist?
    ↓ No
Build: best.onnx → best.onnx.fp16.engine (via TensorRT)
    ↓
Cache engine for future use
    ↓
Ready for inference!
```

### 2. Inference Pipeline
```
Image arrives on ROS topic
    ↓
Convert ROS → OpenCV Mat
    ↓
Letterbox Resize (640x640, store scale/padding)
    ↓
Normalize (/255.0, BGR→RGB)
    ↓
HWC → CHW format
    ↓
Copy to GPU
    ↓
TensorRT Inference
    ↓
Copy output to CPU
    ↓
Parse predictions (N×85 format)
    ↓
Filter by confidence (>0.6)
    ↓
Apply NMS (IoU<0.45)
    ↓
Transform coordinates (letterbox → original)
    ↓
Clamp to image bounds
    ↓
Publish Detection2DArray
```

## Testing Strategy

### Test 1: Engine Creation
```bash
# Provide PyTorch checkpoint
model_path: /path/to/best.pt

# Expected output:
# [INFO] Detected PyTorch checkpoint
# [INFO] Converting PyTorch to ONNX...
# [INFO] Successfully converted to best.onnx
# [INFO] Building TensorRT engine with FP16...
# [INFO] Saving engine to best.onnx.fp16.engine
# [INFO] Engine initialized successfully
```

### Test 2: Inference
```bash
# Publish test image
ros2 topic pub /sensors/camera/image_raw sensor_msgs/msg/Image ...

# Monitor detections
ros2 topic echo /autospeed/detections

# Expected output:
# detections:
#   - bbox: {center: {x: 320, y: 240}, size_x: 100, size_y: 150}
#     results:
#       - hypothesis: {class_id: "0", score: 0.95}
```

### Test 3: Performance
```bash
# Check FPS (printed every 100 frames)
# Expected: 30-50 FPS on decent GPU (with FP16)
```

## Differences from Your Original Python Code

| Python Version | C++ Version |
|----------------|-------------|
| `torch.load()` | PyTorch→ONNX→TensorRT |
| `model.half()` | Precision param: "fp16" |
| `PIL.Image` | `cv::Mat` |
| `transforms.ToTensor()` | Manual preprocessing |
| `ops.nms()` (torchvision) | Custom NMS implementation |
| Python lists | `std::vector` |
| NumPy arrays | Raw float arrays |

## Why This Approach?

### ✅ Advantages
1. **No PyTorch runtime dependency** in C++ (only for conversion)
2. **Faster inference** (TensorRT optimizations)
3. **Smaller memory footprint**
4. **Better integration** with ROS2 ecosystem
5. **Cached engines** = fast startup after first run

### ⚠️ Considerations
1. **First run is slower** (engine building takes 1-2 minutes)
2. **Requires PyTorch** for `.pt` → `.onnx` conversion (one-time)
3. **GPU-specific engines** (can't share between different GPU models)

## Next Steps After Building

1. **Test with sample images** - Verify detections are correct
2. **Tune thresholds** - Adjust `conf_threshold` and `iou_threshold` for your use case
3. **Create visualization node** - Draw boxes on images for debugging
4. **Benchmark performance** - Monitor FPS and latency
5. **Train AutoSpeed model** - Create dataset and train for your specific task

## Questions to Consider

1. **Model Input**: Do you have a PyTorch checkpoint ready? What's the path?
2. **Classes**: How many object classes does your model detect?
3. **Visualization**: Do you want a separate node to draw bounding boxes on images?
4. **Output Format**: Is `vision_msgs::Detection2DArray` suitable, or do you need a custom message?

Let me know if you need help with any of these next steps!

