# AutoSpeed ObjectFinder Module

## Overview

The ObjectFinder module extends AutoSpeed object detection with **tracking**, **distance/velocity estimation**, and **Mobileye RSS safety assessment**. It identifies the **CIPO** (Closest In-Path Object) and calculates safe following distances for autonomous driving applications.

## Architecture

```
GStreamer Stream → AutoSpeed Detection → ObjectFinder Tracking → CIPO + RSS Safety
```

### Key Components

1. **AutoSpeed Detection**: Bounding box object detection (pedestrians, bicycles, cars)
2. **Homography Transform**: Image pixels → world coordinates (meters)
3. **Kalman Filter**: Smooth distance/velocity tracking over time
4. **CIPO Selection**: Priority-based selection of most critical object
5. **RSS Calculation**: Mobileye Responsibility-Sensitive Safety distance formula

## Features

### Distance Estimation
- Uses camera calibration (homography matrix) to convert bbox positions to metric distances
- Object distance calculated from bbox bottom center (ground contact point)
- Accurate for flat road assumption

### Velocity Estimation
- Tracks distance changes over time
- Kalman filter smoothing: `State = [distance, velocity]`
- Velocity in m/s (negative = approaching)

### CIPO (Closest In-Path Object) Selection
Priority score based on:
- **Distance**: Closer objects = higher priority
- **Class**: Pedestrian > Bicycle > Car
- **In-path**: Objects within ego lane boundaries

### RSS Safety Check
Implements Mobileye's formula:

```
d_min = [v_r × ρ + 0.5 × a_max_accel × ρ² + (v_r + ρ × a_max_accel)² / (2 × a_min_brake)]
        - v_f² / (2 × a_max_brake)
```

Where:
- `v_r` = ego vehicle velocity (m/s)
- `v_f` = front vehicle velocity (m/s)
- `ρ` = response time (default: 1.0s)
- `a_max_accel` = max acceleration (default: 2.0 m/s²)
- `a_min_brake` = ego min braking (default: 4.0 m/s²)
- `a_max_brake` = front max braking (default: 6.0 m/s²)

**Safety Status:**
- `SAFE`: `distance ≥ d_min` → maintain or accelerate
- `UNSAFE`: `distance < d_min` → brake required

## Camera Calibration

### Required: Homography Matrix

The homography matrix `H` transforms image pixels to world coordinates (meters):

```yaml
H: [h11, h12, h13,
    h21, h22, h23,
    h31, h32, h33]
```

### Calibration Methods

#### Method 1: OpenCV Homography (Recommended)

```python
import cv2
import numpy as np
import yaml

# Define known correspondences (image pixels → world meters)
src_points = np.array([
    [960, 700],   # Image center bottom (ego position)
    [800, 500],   # Left lane marking at 10m
    [1120, 500],  # Right lane marking at 10m
    [900, 400],   # Left lane at 20m
    # ... more points
], dtype=np.float32)

dst_points = np.array([
    [0, 0],       # Ego position (origin)
    [-1.8, 10],   # Left lane at 10m (lane width = 3.6m)
    [1.8, 10],    # Right lane at 10m
    [-1.8, 20],   # Left lane at 20m
    # ... corresponding world points
], dtype=np.float32)

# Compute homography
H, status = cv2.findHomography(src_points, dst_points)

# Save to YAML
with open('calibration.yaml', 'w') as f:
    yaml.dump({'H': H.flatten().tolist()}, f)
```

#### Method 2: Using Known Lane Markings

Highway lane markings are standardized:
- Lane width: 3.6 meters
- Dashed line length: 3 meters
- Dashed line gap: 9 meters

Mark these in your video frames to get calibration points.

#### Method 3: Simplified Calibration (Quick Testing)

If you don't have precise calibration, use the sample provided:
- Based on typical dashcam setup
- Camera height: ~1.2m
- Accuracy: ±20% (sufficient for testing)

## Build Instructions

```bash
cd VisionPilot/Standalone/AutoSpeed
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### Dependencies
- OpenCV 4.x
- CUDA 11+
- TensorRT 8+
- GStreamer 1.0
- yaml-cpp

## Usage

### Basic Usage

```bash
./build/autospeed_object_finder \
    <video_file> \
    <model_path> \
    <calibration_yaml> \
    <precision> \
    <ego_velocity>
```

### Example

```bash
./build/autospeed_object_finder \
    video.mp4 \
    AutoSpeed_n.onnx \
    calibration_sample.yaml \
    fp16 \
    15.0  # 15 m/s = ~54 km/h
```

### Using Test Script

```bash
# Edit test_object_finder.sh to set your paths
./test_object_finder.sh
```

### Command-Line Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `stream_source` | Video file, RTSP URL, or /dev/videoX | `video.mp4` |
| `model_path` | AutoSpeed model (.pt or .onnx) | `AutoSpeed_n.onnx` |
| `homography_yaml` | Calibration file | `calibration.yaml` |
| `precision` | TensorRT precision | `fp16` or `fp32` |
| `ego_velocity` | Ego speed in m/s | `15.0` (~54 km/h) |
| `realtime` | (Optional) Sync to video FPS | `true` / `false` |
| `measure_latency` | (Optional) Show metrics | `true` / `false` |

## Output

### CIPO Information (Printed to Console)

```
========== CIPO DETECTED (Frame 123) ==========
Track ID: 5 | Class: 3
Distance: 25.30 m
Velocity: 12.50 m/s
Lateral Offset: -0.80 m
Time-to-Collision: 10.12 s
RSS Safe Distance: 18.75 m
Status: SAFE ✓
Priority Score: 15.32
========================================
```

### Latency Metrics (Every 100 frames)

```
========================================
Frames processed: 100
Pipeline Latencies:
  1. Capture:   2.50 ms
  2. Inference: 8.30 ms
  3. Tracking:  1.20 ms
Total tracked objects: 100
========================================
```

## Interpretation

### Ego Vehicle Control Logic

Based on CIPO information, implement these control scenarios:

#### Scenario 1: No CIPO
```cpp
if (!cipo.exists) {
    // No object in path
    action = ACCELERATE_TO_SPEED_LIMIT;
}
```

#### Scenario 2: Safe Distance
```cpp
else if (cipo.is_safe && cipo.distance_m > cipo.safe_distance_rss * 1.5) {
    // Safe distance maintained
    action = MATCH_CIPO_SPEED;
}
```

#### Scenario 3: Unsafe - Brake Required
```cpp
else if (!cipo.is_safe) {
    // Distance < RSS safe distance
    action = BRAKE;
    brake_intensity = calculateBrakeIntensity(cipo.distance_m, cipo.safe_distance_rss);
}
```

### RSS Parameters Tuning

Default conservative values:
```cpp
RSSParameters rss_params;
rss_params.response_time = 1.0f;      // 1 second
rss_params.max_accel = 2.0f;          // 2 m/s²
rss_params.min_brake_ego = 4.0f;      // 4 m/s²
rss_params.max_brake_front = 6.0f;    // 6 m/s²
```

**Adjust for your application:**
- **Highway**: Increase response time to 1.2-1.5s (higher speeds)
- **Urban**: Decrease to 0.8s (lower speeds, better control)
- **Aggressive**: Increase `min_brake_ego` to 5-6 m/s²
- **Conservative**: Decrease `max_brake_front` to 4-5 m/s² (assume slower front vehicle braking)

## Performance

### Latency Breakdown (Typical)

| Stage | FP32 | FP16 |
|-------|------|------|
| Capture | 2-3 ms | 2-3 ms |
| Inference | 12-15 ms | 6-9 ms |
| Tracking | 1-2 ms | 1-2 ms |
| **Total** | **15-20 ms** | **9-14 ms** |

**Throughput:** ~70 FPS (FP16) / ~50 FPS (FP32) on typical GPU (e.g., RTX 3060)

## Troubleshooting

### Issue: Incorrect Distance Estimation

**Solution:**
1. Check calibration accuracy
2. Verify `H` matrix is correct
3. Test with known objects at measured distances
4. Use PathFinder's calibration method

### Issue: Noisy Velocity

**Solution:**
1. Increase Kalman process noise covariance (in `object_finder.cpp`)
2. Ensure stable tracking (IoU threshold)
3. Increase `frames_tracked` threshold for CIPO selection

### Issue: No CIPO Detected

**Solution:**
1. Check ego lane width parameter (`ego_lane_width_` = 3.6m default)
2. Adjust class priorities in `getClassPriority()`
3. Verify detections are in front (positive Y world coordinate)

## Future Enhancements

- [ ] Multi-object tracking (SORT/DeepSORT algorithm)
- [ ] Lane detection integration for precise in-path scoring
- [ ] Ego-motion estimation (when CAN bus unavailable)
- [ ] Curve/hill compensation for homography
- [ ] Real-time RSS parameter adaptation

## References

1. **Mobileye RSS**: https://www.mobileye.com/technology/responsibility-sensitive-safety/
2. **PathFinder Module**: `../../../PathFinder/README.md`
3. **AutoSpeed Detection**: `../README.md`

