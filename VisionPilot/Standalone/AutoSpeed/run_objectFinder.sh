#!/bin/bash
# Quick test script for AutoSpeed standalone inference

# Suppress GStreamer warnings
export GST_DEBUG=1

VIDEO_PATH="/home/pranavdoma/Downloads/autoware.privately-owned-vehicles/VisionPilot/ROS2/data/segment_video.mp4"
MODEL_PATH="/home/pranavdoma/Downloads/autoware.privately-owned-vehicles/VisionPilot/ROS2/data/models/AutoSpeed_n.onnx"
PRECISION="fp16"
HOMOGRAPHY_YAML="/home/pranavdoma/Downloads/autoware.privately-owned-vehicles/VisionPilot/Calibration/homography.yaml"
REALTIME="true"           # Real-time playback (matches video FPS)
MEASURE_LATENCY="true"    # Enable latency metrics
ENABLE_VIZ="true"         # Enable visualization (set to "false" for headless mode)
SAVE_VIDEO="false"        # Enable saving output video (requires ENABLE_VIZ=true)
OUTPUT_VIDEO="output_tracking.mp4" # Output video file path


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
echo "Video: $VIDEO_PATH"
echo "Model: $MODEL_PATH"
echo "Precision: $PRECISION"
echo "Homography: $HOMOGRAPHY_YAML"
echo ""
echo "Press 'q' in the video window to quit"
echo "=========================================="
echo ""

./build/autospeed_infer_stream "$VIDEO_PATH" "$MODEL_PATH" "$PRECISION" "$HOMOGRAPHY_YAML" "$REALTIME" "$MEASURE_LATENCY" "$ENABLE_VIZ" "$SAVE_VIDEO" "$OUTPUT_VIDEO"

