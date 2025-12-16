#!/bin/bash

# Enable X11 forwarding for visualization
xhost +

# Run the container
docker run -it --rm \
    -p 6080:6080 \
    -v "$(pwd)/model-weights:/autoware/visionpilot/model-weights" \
    -v "$(pwd)/test:/autoware/visionpilot/test" \
    ghcr.io/autowarefoundation/visionpilot:latest \
    python3 /autoware/Models/visualizations/Scene3D/image_visualization.py -p /autoware/visionpilot/model-weights/scene3d.pth -i /autoware/visionpilot/test/image.jpg
