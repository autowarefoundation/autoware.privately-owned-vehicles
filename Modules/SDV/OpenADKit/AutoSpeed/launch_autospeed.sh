#!/bin/bash
# Containerized AutoSpeed(ObjectFinder) standalone inference with ONNX Runtime

# Enable X11 forwarding for visualization
xhost +

# Run the container
docker run -it --rm \
    -p 6080:6080 \
    -v "$PWD"/model-weights:/autoware/model-weights \
    -v "$PWD"/launch:/autoware/launch \
    -v "$PWD"/../Test:/autoware/test \
    ghcr.io/autowarefoundation/visionpilot:latest \
    /autoware/launch/run_objectFinder.sh
