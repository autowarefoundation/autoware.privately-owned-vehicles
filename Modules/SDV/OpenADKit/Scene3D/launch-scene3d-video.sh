#!/bin/bash

# Enable X11 forwarding for visualization
xhost +

# Run the container
docker run -it --rm \
    -p 6080:6080 \
    -v "$(pwd)/model-weights:/autoware/visionpilot/model-weights" \
    -v "$(pwd)/test:/autoware/visionpilot/test" \
    ghcr.io/autowarefoundation/visionpilot:latest \
    python3 /autoware/Models/visualizations/Scene3D/video_visualization.py -v -p /autoware/visionpilot/model-weights/scene3d.pth -i /autoware/visionpilot/test/traffic-driving.mp4 -o /autoware/visionpilot/test/output_scene3d.avi
