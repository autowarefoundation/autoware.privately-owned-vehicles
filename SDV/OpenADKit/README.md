# VisionPilot - OpenADKit Demo

Containerized Scene3D Demo for VisionPilot.

## Prerequisites

- Docker

- Download the [Scene3D PyTorch model weights](https://github.com/autowarefoundation/autoware.privately-owned-vehicles/tree/main/Models#scene3d---monocular-depth-estimation) and place it in the `model-weights` directory named as `scene3d.pth`.

## Build the Docker image

```bash
docker build -t visionpilot -f docker/Dockerfile ../..
```

## Usage

```bash
./launch-scene3d.sh
```

## Output

The output will be displayed in a new window that shows monocular depth estimation of the input image.
