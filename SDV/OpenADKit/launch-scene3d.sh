#!/bin/bash

# Enable X11 forwarding for visualization
xhost +

# Run the container
docker run -it --rm \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$(pwd)/model-weights:/visionpilot/model-weights" \
    -v "$(pwd)/test:/visionpilot/test" \
    visionpilot:latest \
    python3 /visionpilot/Models/visualizations/Scene3D/image_visualization.py -p /visionpilot/model-weights/scene3d.pth -i /visionpilot/test/image.jpg
