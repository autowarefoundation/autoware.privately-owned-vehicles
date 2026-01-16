#!/bin/bash

# Enable X11 forwarding for visualization
xhost +

# Run the container
docker run -it --rm \
    -p 6080:6080 \
    -v "$PWD"/model-weights:/autoware/model-weights \
    -v "$PWD"/test:/autoware/test \
    visionpilot:latest \
    python3 /autoware/Models/visualizations/SceneSeg/video_visualization.py -v -p /autoware/model-weights/sceneseg.pth -i /autoware/test/traffic-driving.mp4 -o /autoware/test/output_sceneseg.avi
