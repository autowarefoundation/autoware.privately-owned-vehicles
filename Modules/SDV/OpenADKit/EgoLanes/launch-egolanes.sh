#!/bin/bash

# Run the container
docker run -it --rm \
    -p 6080:6080 \
    -v "$PWD"/model-weights:/autoware/model-weights \
    -v "$PWD"/test:/autoware/test \
    ghcr.io/autowarefoundation/visionpilot:latest \
    python3 /autoware/Models/visualizations/EgoLanes/video_visualization.py -p /autoware/model-weights/egolanes.pth -i /autoware/test/traffic-driving.mp4 -o /autoware/test/output_egolanes.avi
