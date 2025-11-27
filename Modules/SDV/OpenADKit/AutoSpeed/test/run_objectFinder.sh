#!/bin/bash
# Containerized AutoSpeed standalone inference with ONNX Runtime

# Suppress GStreamer warnings
export GST_DEBUG=1

# ===== Required Parameters =====
VIDEO_PATH="/autoware/test/traffic-driving.mp4"
MODEL_PATH="/autoware/model-weights/autospeed.onnx"
PROVIDER="cpu"       # Execution provider: 'cpu' or 'tensorrt'
PRECISION="fp32"          # Precision: 'fp32' or 'fp16' (for TensorRT)
HOMOGRAPHY_YAML="/autoware/VisionPilot/Standalone/AutoSpeed/homography.yaml"

# ===== ONNX Runtime Options =====
DEVICE_ID="0"             # GPU device ID (TensorRT only)
CACHE_DIR="/autoware/trt_cache"   # TensorRT engine cache directory

# ===== Pipeline Options =====
REALTIME="true"           # Real-time playback (matches video FPS)
MEASURE_LATENCY="false"    # Enable latency metrics
ENABLE_VIZ="true"         # Enable visualization (set to "false" for headless mode)
SAVE_VIDEO="false"         # Enable saving output video (requires ENABLE_VIZ=true)
OUTPUT_VIDEO="output_tracking_${PRECISION}_${PROVIDER}.mp4" # Output video file path


if [ ! -f "$VIDEO_PATH" ]; then
    echo "Error: Video file not found: $VIDEO_PATH"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$HOMOGRAPHY_YAML" ]; then
    echo "Error: Homography calibration file not found: $HOMOGRAPHY_YAML"
    exit 1
fi

echo "Starting AutoSpeed standalone inference with tracking..."
echo "========================================"
echo "Video: $VIDEO_PATH"
echo "Model: $MODEL_PATH"
echo "Provider: $PROVIDER"
echo "Precision: $PRECISION"
echo "Homography: $HOMOGRAPHY_YAML"
if [ "$PROVIDER" == "tensorrt" ]; then
    echo "Device ID: $DEVICE_ID"
    echo "Cache Dir: $CACHE_DIR"
fi
echo "========================================"
echo "Visualization: $ENABLE_VIZ"
echo "Save Video: $SAVE_VIDEO"
if [ "$SAVE_VIDEO" == "true" ]; then
    echo "Output: $OUTPUT_VIDEO"
fi
echo "========================================"
echo ""
if [ "$ENABLE_VIZ" == "true" ]; then
    echo "Press 'q' in the video window to quit"
else
    echo "Running in headless mode (console output only)"
fi
echo ""

# Make sure ONNXRUNTIME_ROOT is set
if [ -z "$ONNXRUNTIME_ROOT" ]; then
    echo "Warning: ONNXRUNTIME_ROOT not set. Please export it:"
    echo "  export ONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-gpu-X.X.X"
    exit 1
fi

/autoware/autospeed_infer_stream "$VIDEO_PATH" "$MODEL_PATH" "$PROVIDER" "$PRECISION" "$HOMOGRAPHY_YAML" "$DEVICE_ID" "$CACHE_DIR" "$REALTIME" "$MEASURE_LATENCY" "$ENABLE_VIZ" "$SAVE_VIDEO" "$OUTPUT_VIDEO"
