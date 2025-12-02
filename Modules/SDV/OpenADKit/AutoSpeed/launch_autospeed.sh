#!/bin/bash
# Containerized AutoSpeed(ObjectFinder) standalone inference with ONNX Runtime

# Enable X11 forwarding for visualization
xhost +

# Run the container
docker run -it --rm \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$(pwd)/model-weights:/autoware/model-weights" \
    -v "$(pwd)/test:/autoware/test" \
    ghcr.io/autowarefoundation/visionpilot:latest \
    /autoware/test/run_objectFinder.sh
