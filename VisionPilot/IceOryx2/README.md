# VisionPilot IceOryx2 Pipeline

Zero-copy IPC-based vision pipeline using **iceoryx2** for high-performance distributed inference and tracking.


### **Services (iceoryx2)**

| Service Name              | Data Type      | Publisher         | Subscribers           |
|---------------------------|----------------|-------------------|-----------------------|
| `VisionPilot/RawFrames`   | `RawFrame`     | `frame_node`      | `inference_node`, `viz_node` |
| `VisionPilot/CIPO`        | `CIPOMessage`  | `inference_node`  | `viz_node`, (external) |

---

## Build

### **Prerequisites**

1. **iceoryx2** (C++ bindings)
   ```bash
   # Install from: https://github.com/eclipse-iceoryx/iceoryx2
   # Or via package manager if available
   ```

2. **ONNX Runtime**
   ```bash
   export ONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-gpu-X.X.X
   ```

3. **OpenCV** (with GStreamer support)
   ```bash
   sudo apt install libopencv-dev
   ```

4. **GStreamer**
   ```bash
   sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
   ```

5. **cuDNN** (for TensorRT provider)
   ```bash
   # Download from NVIDIA, then:
   export LD_LIBRARY_PATH=/path/to/cudnn/lib:$LD_LIBRARY_PATH
   ```

### **Build Steps**

```bash
cd VisionPilot/IceOryx2
mkdir build
cd build
cmake ..
make -j$(nproc)
```

**Output:**
- `build/frame_node` - Frame publisher
- `build/inference_node` - Inference + tracking
- `build/viz_node` - Visualization (optional)

---

## Usage

### **Option 1: Launch Script (Recommended)**

1. **Edit `run_pipeline.sh`** with your paths:
   ```bash
   VIDEO_PATH="/path/to/video.mp4"
   MODEL_PATH="/path/to/model.onnx"
   HOMOGRAPHY_YAML="/path/to/homography.yaml"
   PROVIDER="tensorrt"       # or "cpu"
   PRECISION="fp16"          # or "fp32"
   ENABLE_VIZ="true"         # or "false" for headless
   ```

2. **Run:**
   ```bash
   bash run_pipeline.sh
   ```

3. **Stop:**
   Press `Ctrl+C` (stops all nodes gracefully)

---

### **Option 2: Manual Launch (Advanced)**

**Terminal 1: Frame Publisher**
```bash
./build/frame_node /path/to/video.mp4 true
```

**Terminal 2: Inference + Tracking**
```bash
export ONNXRUNTIME_ROOT=/path/to/onnxruntime
./build/inference_node \
    /path/to/model.onnx \
    tensorrt \
    fp16 \
    /path/to/homography.yaml \
    0 \
    ./trt_cache
```

**Terminal 3: Visualization (Optional)**
```bash
./build/viz_node                           # Display only
./build/viz_node --save output.mp4         # Save video
```

---

## Features

### **frame_node**
- **Input:** Video file or camera device (GStreamer)
- **Output:** `RawFrame` (1920x1280x3, ~7.3 MB per frame)
- **Features:**
  - Realtime or max-speed playback
  - Zero-copy publishing via shared memory
  - FPS statistics

### **inference_node**
- **Input:** `RawFrame` (zero-copy read)
- **Output:** `CIPOMessage` (main CIPO data, ~120 bytes)
- **Features:**
  - ONNX Runtime (CPU/TensorRT providers)
  - Multi-object tracking (Kalman filter)
  - CIPO selection (Level 1 > Level 2)
  - Cut-in detection (ORB feature matching)
  - Console output for pub/sub integration

### **viz_node**
- **Input:** `RawFrame` + `CIPOMessage`
- **Output:** OpenCV window (optional video saving)
- **Features:**
  - Detachable (can start/stop independently)
  - Semi-transparent bounding boxes
  - Distance & velocity labels
  - Cut-in warnings
  - Frame synchronization by `frame_id`

---

## Configuration

### **Realtime vs. Benchmark Mode**
```bash
./build/frame_node video.mp4 true   # Realtime (matches video FPS)
./build/frame_node video.mp4 false  # Benchmark (max speed)
```

### **CPU vs. TensorRT**
```bash
# CPU (slower, no GPU required)
./build/inference_node model.onnx cpu fp32 homography.yaml 0 ./cache

# TensorRT (faster, requires CUDA + cuDNN)
./build/inference_node model.onnx tensorrt fp16 homography.yaml 0 ./trt_cache
```

### **Headless Mode (No Visualization)**
```bash
# Don't launch viz_node
# inference_node will still output main_CIPO to console
```

---

## Data Structures

### **RawFrame** (7.3 MB)
```cpp
struct RawFrame {
    uint64_t frame_id;       // Sequential ID
    uint64_t timestamp_ns;   // Capture time
    uint32_t width, height;  // 1920x1280
    uint8_t data[...];       // BGR image
    bool is_valid;
};
```

### **CIPOMessage** (~120 bytes)
```cpp
struct CIPOMessage {
    uint64_t frame_id;       // Links to RawFrame
    bool exists;             // Main CIPO detected?
    int32_t track_id;        // Tracking ID
    int32_t class_id;        // CIPO level (1, 2, 3)
    float distance_m;        // Distance (meters)
    float velocity_ms;       // Velocity (m/s)
    float bbox[4];           // Bounding box
    bool cut_in_detected;    // Event flag
    bool kalman_reset;       // Event flag
};
```

---



