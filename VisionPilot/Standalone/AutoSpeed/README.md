# AutoSpeed Standalone Inference

Multi-threaded GStreamer + TensorRT inference for real-time object detection.

## Features

- **Multi-threaded Architecture**: Separate threads for capture, inference, and display
- **Zero Bottleneck**: Inference never waits for display
- **Real-time Metrics**: Separate FPS measurements for each pipeline stage
- **GStreamer Support**: RTSP streams, USB cameras, video files

## Architecture

```
Capture Thread → Queue → Inference Thread → Queue → Display Thread
                         (HIGH PRIORITY)           (LOW PRIORITY)
                         
Latency 1: Frame → Preprocess → Inference (~4-6ms)
Latency 2: Draw boxes → cv::imshow (~10-20ms)
```

**Key Design:** Inference thread processes frames independently of display, achieving maximum throughput.

## Build

```bash
cd VisionPilot/Standalone/AutoSpeed
mkdir build && cd build
cmake .. -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4 \
         -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Usage

### RTSP Stream
```bash
./autospeed_infer_stream rtsp://192.168.1.10:8554/stream /path/to/model.onnx fp16
```

### USB Camera
```bash
./autospeed_infer_stream /dev/video0 /path/to/model.onnx fp32
```

### Video File
```bash
./autospeed_infer_stream /path/to/video.mp4 /path/to/model.onnx fp16
```

## Metrics Output

Every 2 seconds, the application prints:

```
========================================
CAPTURE FPS:    30.00
INFERENCE FPS:  160.00 (avg latency: 4.25 ms)
DISPLAY FPS:    30.00 (avg latency: 12.50 ms)
========================================
```

**Interpretation:**
- **CAPTURE FPS**: Stream framerate (limited by source)
- **INFERENCE FPS**: Model throughput (your hardware capability)
- **DISPLAY FPS**: Visualization rate (can drop frames)

**Inference FPS > Capture FPS** = Your model is fast enough for real-time!

## Controls

- Press `q` to quit

## Dependencies

- GStreamer 1.0 (with app plugin)
- OpenCV 4.x
- CUDA 11.x or 12.x
- TensorRT 8.x or 10.x

## Notes

- **Inference Priority**: Set to high to ensure real-time performance
- **Display Drops Frames**: If inference is faster than 60 FPS, display will drop frames (human eyes can't see >60 FPS anyway)
- **Queue Sizes**: Automatically managed (drops oldest frame if full)

