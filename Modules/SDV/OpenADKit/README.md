# VisionPilot - OpenADKit Workloads

Containerized VisionPilot workloads for Open AD Kit deployments.

## Prerequisites

- Docker

- Download the [Traffic Driving video](https://drive.google.com/file/d/1_mFCpsKkBrotVUiv_OIZi1B6Fd3UXUG3/view?usp=drive_link) to use as input for the workloads and place it in the `Test` directory with the name `traffic-driving.mp4`. Below script will do this for you:

    ```bash
    mkdir -p Test
    curl "https://drive.usercontent.google.com/download?id=1_mFCpsKkBrotVUiv_OIZi1B6Fd3UXUG3&confirm=xxx" -o Test/traffic-driving.mp4
    ```

### Building the Docker image from scratch

The **visionpilot** container image is available on [GHCR](https://github.com/orgs/autowarefoundation/packages/container/package/visionpilot), but if you want to build it locally, you can do so by running the build command:

```bash
# Run from project root
docker build -t visionpilot -f Modules/SDV/OpenADKit/Docker/Dockerfile .
```
