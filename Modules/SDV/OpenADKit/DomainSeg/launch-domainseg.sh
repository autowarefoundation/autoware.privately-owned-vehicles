#!/bin/bash
# Containerized DomainSeg standalone inference with ONNX Runtime

# Run the container
docker run -it --rm \
    -p 6080:6080 \
    -v "$PWD"/model-weights:/autoware/model-weights \
    -v "$PWD"/../Test:/autoware/test \
    visionpilot:latest \
    python3 /autoware/Models/visualizations/DomainSeg/video_visualization.py -v -p /autoware/model-weights/domainseg.pth -i /autoware/test/traffic-driving.mp4 -o /autoware/test/output_domainseg.avi
