# VisionPilot Common - Framework-Agnostic Core Engines

This directory contains the core AI processing engines that are completely independent of any middleware framework. These engines provide the fundamental inference, visualization, and sensor processing capabilities that can be wrapped by any middleware implementation (ROS2, Zenoh, etc.).

## Directory Structure

```
common/
├── backends/                 # Inference engine implementations
│   ├── onnx_runtime_backend.cpp
│   ├── tensorrt_backend.cpp
│   └── (future custom backends)
├── visualizers/             # Visualization processing engines
│   ├── masks_visualization_engine.cpp
│   ├── depth_visualization_engine.cpp
│   └── (future visualization engines)
├── sensors/                 # Input processing engines
│   ├── video_publisher_engine.cpp
│   └── (future sensor engines)
├── include/                 # Header files
│   ├── inference_backend_base.hpp
│   ├── onnx_runtime_backend.hpp
│   ├── tensorrt_backend.hpp
│   ├── masks_visualization_engine.hpp
│   ├── depth_visualization_engine.hpp
│   └── video_publisher_engine.hpp
└── README.md               # This file
```

## Core Engines

### Inference Backends (`/backends/`)

#### Base Interface (`inference_backend_base.hpp`)

Abstract base class defining the common interface for all inference backends:

```cpp
class InferenceBackendBase {
public:
    virtual bool initialize(const std::string& model_path, const std::string& precision) = 0;
    virtual bool getRawOutput(cv::Mat& output, const cv::Size& target_size, 
                             const std::string& model_type = "segmentation") = 0;
    virtual bool preprocess(cv::Mat& input_image) = 0;
};
```

#### ONNX Runtime Backend (`onnx_runtime_backend.cpp`)

- **Purpose**: CPU and CUDA inference using Microsoft ONNX Runtime
- **Features**:
  - Automatic provider selection (CUDA, CPU)
  - Dynamic input tensor handling
  - HWC to CHW conversion with proper normalization
  - Model-type aware post-processing
- **Post-processing Logic**:
  - **Segmentation**: Argmax (multi-class) or threshold (binary) → CV_8UC1 class masks
  - **Depth**: Raw float output → CV_32FC1 depth maps
- **Preprocessing**: Normalization with Scene Seg values (mean: [0.406, 0.456, 0.485], std: [0.225, 0.224, 0.229])

#### TensorRT Backend (`tensorrt_backend.cpp`)

- **Purpose**: Optimized NVIDIA GPU inference with TensorRT
- **Features**:
  - Engine file caching and loading
  - Dynamic batch size support
  - FP16/FP32 precision control
  - Zero-copy GPU memory management
- **Performance**: Typically 2-3x faster than ONNX Runtime on NVIDIA GPUs
- **Post-processing**: Identical logic to ONNX Runtime backend

### Visualization Engines (`/visualizers/`)

#### Masks Visualization Engine (`masks_visualization_engine.cpp`)

- **Purpose**: Segmentation mask colorization and blending
- **Supported Types**:
  - **Scene Segmentation**: Black background (0,0,0), Red foreground (255,0,0)
  - **Domain Segmentation**: Orange road (255,93,61), Blue off-road (28,148,255)
- **Input**: CV_8UC1 class ID masks from inference backends
- **Output**: CV_8UC3 colorized and blended visualization
- **Features**:
  - Efficient pixel-wise color mapping
  - Alpha blending with original image
  - Exact color reproduction from original implementations

#### Depth Visualization Engine (`depth_visualization_engine.cpp`)

- **Purpose**: Depth map visualization and normalization
- **Input**: CV_32FC1 raw depth values from inference backends
- **Output**: CV_8UC3 colorized depth maps using OpenCV color maps
- **Features**:
  - Min-max normalization to [0, 255] range
  - Configurable OpenCV color maps (COLORMAP_JET default)
  - Efficient depth processing for real-time visualization

### Sensor Engines (`/sensors/`)

#### Video Publisher Engine (`video_publisher_engine.cpp`)

- **Purpose**: Framework-agnostic video input processing
- **Input Sources**: Video files, camera streams, image sequences
- **Features**:
  - OpenCV VideoCapture integration
  - Configurable frame rate control
  - Frame buffering and timing management
  - Error handling for invalid sources
- **Output**: CV_8UC3 BGR frames ready for middleware publishing

## Design Principles

### Framework Independence

- **No Middleware Dependencies**: All engines use only standard C++ libraries and OpenCV
- **Clean Interfaces**: Well-defined APIs that can be wrapped by any middleware
- **Configuration Driven**: Behavior controlled by parameters, not hardcoded values

### Extensibility

- **Plugin Architecture**: Easy to add new backends by implementing base interface
- **Configurable Processing**: Model type parameter controls post-processing behavior
- **Future Proof**: Designed to accommodate new AI tasks and model types

## Integration Guide

### Adding New Inference Backend

1. Inherit from `InferenceBackendBase`
2. Implement required virtual methods
3. Handle model-type specific post-processing
4. Add to middleware wrapper's backend selection logic

### Adding New Visualization Engine

1. Create new engine class with `visualize()` method
2. Implement task-specific color mapping and rendering
3. Add to middleware wrapper's visualization node selection

### Adding New Sensor Engine

1. Implement input source handling (camera, network stream, etc.)
2. Provide standard CV_8UC3 output format
3. Include frame rate and timing management
4. Add to middleware wrapper's sensor node options

### Visualization Engines  

- **Masks Visualization**: ~2-5ms per frame (efficient color mapping)
- **Depth Visualization**: ~5-10ms per frame (normalization + color map)

## Dependencies

### Required Libraries

- **OpenCV 4.x**: Image processing and visualization
- **ONNX Runtime 1.15+**: ONNX model inference
- **TensorRT 8.x**: NVIDIA GPU optimization (optional)
- **CUDA 11.8+**: GPU acceleration (optional)

### Build Requirements

- **C++17**: Modern C++ features and standard library
- **CMake 3.16+**: Build system integration
- **GCC/Clang**: Compiler with C++17 support

The common engines are designed to be the stable, high-performance core that remains unchanged as new middleware implementations are added to VisionPilot.
