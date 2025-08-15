# VisionPilot - Middleware-Agnostic AI Vision Framework

A modular, hardware and middleware-independent vision processing framework designed for autonomous vehicle perception tasks. VisionPilot provides a unified API for AI inference and visualization that can be deployed across different middleware systems.

## Architecture Philosophy

VisionPilot separates core AI processing from middleware-specific implementations, enabling seamless deployment across different robotic frameworks:

```
┌─────────────────────────────────────────────────────────┐
│                    MIDDLEWARE LAYER                     │
├─────────────────────┬───────────────────┬───────────────┤
│       ROS2          │      Zenoh        │    Future     │
│   Implementation    │  Implementation   │ Middlewares   │
│                     │                   │               │
│   ┌─────────────┐   │  ┌─────────────┐  │               │
│   │ ROS2 Nodes  │   │  │ Zenoh Nodes │  │      ...      │
│   │ & Topics    │   │  │ & Messages  │  │               │
│   └─────────────┘   │  └─────────────┘  │               │
└─────────────────────┴───────────────────┴───────────────┘
                       │                   
┌─────────────────────────────────────────────────────────┐
│                    COMMON LAYER                         │
│              (Framework Independent)                    │
├─────────────────────┬───────────────────┬───────────────┤
│  Inference Backends │ Visualization     │ Sensor Input  │
│                     │ Engines           │ Processing    │
│  • ONNX Runtime     │                   │               │
│  • TensorRT         │ • Segmentation    │ • Video       │
│  • Custom Backends  │ • Depth Maps      │ • Camera      │
│                     │ • Point Clouds    │ • Streaming   │
└─────────────────────┴───────────────────┴───────────────┘
```

## Key Benefits

### Middleware Independence

- **Write Once, Deploy Everywhere**: Core AI logic works across ROS2, Zenoh, and future middleware
- **No Vendor Lock-in**: Switch between middleware systems without changing AI code
- **Future Proof**: Add new middleware implementations without touching core engines

### Modular Design

- **Pluggable Backends**: Support for ONNX Runtime, TensorRT, and custom inference engines
- **Configurable Pipelines**: YAML-driven configuration for models, topics, and parameters
- **Independent Components**: Sensors, inference, and visualization can run independently
- **Performance Monitoring**: Built-in latency and FPS measurements
- **Concurrent Execution**: Multiple AI pipelines running simultaneously
- **Resource Efficient**: Shared common engines reduce memory footprint

## Current Implementations

### ROS2 (`/ROS2/`)

Production-ready ROS2 nodes providing:

- Native ROS2 topics and parameters
- Launch file orchestration
- Standard ROS2 message types
- Component-based node architecture

### Zenoh (`/Zenoh/`)

The Zenoh implementation providing:

- Edge computing deployments
- Low-latency communication
- Distributed AI processing
- Cloud-edge hybrid architectures

### Common Core (`/common/`)

Framework-agnostic engines providing:

- AI inference backends (ONNX Runtime, TensorRT)
- Visualization rendering (segmentation masks, depth maps)
- Sensor input processing (video streams, camera feeds)

## Quick Start

### ROS2 Implementation

```bash
cd VisionPilot/ROS2
colcon build --packages-select sensors models visualization
source install/setup.bash

# Run complete pipeline
ros2 launch models run_pipeline.launch.py \
  pipeline:=scene_seg \
  video_path:="data/your_video.mp4"
```

### Custom Middleware Integration

To add a new middleware implementation:

1. Create `/YourMiddleware/` directory
2. Implement thin wrapper nodes around common engines
3. Handle middleware-specific message passing
4. Leverage existing common backends and visualizers

## Repository Structure

```
VisionPilot/
├── common/         # Framework-agnostic core engines
├── ROS2/           # ROS2-specific implementation  
├── Zenoh/          # Future Zenoh implementation
└── README.md       # This file
```

For technical details on core engines, see `common/README.md`
For ROS2-specific usage, see `ROS2/README.md`
For Zenoh-specific usage, see `Zenoh/README.md`
