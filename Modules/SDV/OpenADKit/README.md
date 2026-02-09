# VisionPilot - Open AD Kit Demos

VisionPilot demos with Open AD Kit containers.

## Prerequisites

- Docker

- Download the [Traffic Driving video](https://drive.google.com/file/d/1_mFCpsKkBrotVUiv_OIZi1B6Fd3UXUG3/view?usp=drive_link) to use as input for the demos and place it in the `Test` directory with the name `traffic-driving.mp4`. Below script will do this for you:

    ```bash
    mkdir -p Test
    curl "https://drive.usercontent.google.com/download?id=1_mFCpsKkBrotVUiv_OIZi1B6Fd3UXUG3&confirm=xxx" -o Test/traffic-driving.mp4
    ```

### Building the Docker image from scratch

The **visionpilot** container image is automatically pulled from [GHCR](https://github.com/orgs/autowarefoundation/packages/container/package/visionpilot) when running demos. To build it locally instead, **run from the project root**:

```bash
# Build for x64 with ONNX Runtime 1.22.0
docker build -t visionpilot -f Modules/SDV/OpenADKit/Docker/Dockerfile . --build-arg ARCH=x64 --build-arg ONNXRUNTIME_VERSION=1.22.0

# Build for ARM64 with ONNX Runtime 1.22.0
docker build -t visionpilot -f Modules/SDV/OpenADKit/Docker/Dockerfile . --build-arg ARCH=arm64 --build-arg ONNXRUNTIME_VERSION=1.22.0
```
