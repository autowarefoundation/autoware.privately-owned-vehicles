#!/bin/bash

# Test script for AutoSpeed ObjectFinder (CIPO tracking + RSS safety)
# This script runs the infer_stream2 pipeline with object tracking and distance/velocity estimation

# Suppress GStreamer debug messages
export GST_DEBUG=1

# Configuration
VIDEO_PATH="/home/pranavdoma/Downloads/autoware.privately-owned-vehicles/VisionPilot/ROS2/data/mumbai.mp4"
MODEL_PATH="/home/pranavdoma/Downloads/autoware.privately-owned-vehicles/VisionPilot/ROS2/data/models/AutoSpeed_n.onnx"
CALIBRATION_PATH="./calibration_sample.yaml"
PRECISION="fp16"           # fp16 or fp32
EGO_VELOCITY="15.0"        # Ego vehicle speed in m/s (~54 km/h)
REALTIME="true"            # Real-time playback (matches video FPS)
MEASURE_LATENCY="true"     # Enable latency metrics

echo "====================================================="
echo "  AutoSpeed ObjectFinder Test"
echo "====================================================="
echo "Video: $VIDEO_PATH"
echo "Model: $MODEL_PATH"
echo "Calibration: $CALIBRATION_PATH"
echo "Precision: $PRECISION"
echo "Ego velocity: $EGO_VELOCITY m/s (~$(echo "$EGO_VELOCITY * 3.6" | bc) km/h)"
echo "Realtime: $REALTIME"
echo "Measure latency: $MEASURE_LATENCY"
echo "====================================================="
echo ""

# Run the ObjectFinder pipeline
./build/autospeed_object_finder "$VIDEO_PATH" "$MODEL_PATH" "$CALIBRATION_PATH" "$PRECISION" "$EGO_VELOCITY" "$REALTIME" "$MEASURE_LATENCY"

echo ""
echo "====================================================="
echo "  Test completed"
echo "====================================================="

