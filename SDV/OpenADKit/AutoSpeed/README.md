# VisionPilot/AutoSpeed - OpenADKit Demo

Containerized AutoSpeed Demo for VisionPilot.

## Prerequisites

- Docker

- Download the [AutoSpeed ONNX model weights](https://drive.google.com/file/d/1Zhe8uXPbrPr8cvcwHkl1Hv0877HHbxbB/view?usp=drive_link) and place it in the `model-weights` directory with the name `autospeed.onnx`.

    ```bash
    mkdir -p model-weights
    curl "https://drive.usercontent.google.com/download?id=1Zhe8uXPbrPr8cvcwHkl1Hv0877HHbxbB&confirm=xxx" -o model-weights/autospeed.onnx
    ```

- Download the [Homography calibration file](https://drive.google.com/file/d/124Qb_7s2JG-aH_b3LxTQI_YrHtYrO6uO/view?usp=drive_link) and place it in the `model-weights` directory with the name `homography.yaml`.

    ```bash
    mkdir -p model-weights
    curl "https://drive.usercontent.google.com/download?id=124Qb_7s2JG-aH_b3LxTQI_YrHtYrO6uO&confirm=xxx" -o model-weights/homography.yaml
    ```

- Download the [Free Lane Driving video](https://drive.google.com/file/d/1QP9iboetxSHsmyQWOmrzOiqZWVCB-74C/view?usp=drive_link) and place it in the `test` directory with the name `free-lane-driving.mp4`.

    ```bash
    mkdir -p test
    curl "https://drive.usercontent.google.com/download?id=1QP9iboetxSHsmyQWOmrzOiqZWVCB-74C&confirm=xxx" -o test/free-lane-driving.mp4
    ```

- Download the [Traffic Driving video](https://drive.google.com/file/d/1_mFCpsKkBrotVUiv_OIZi1B6Fd3UXUG3/view?usp=drive_link) and place it in the `test` directory with the name `traffic-driving.mp4`.

    ```bash
    mkdir -p test
    curl "https://drive.usercontent.google.com/download?id=1_mFCpsKkBrotVUiv_OIZi1B6Fd3UXUG3&confirm=xxx" -o test/traffic-driving.mp4
    ```

## Usage

```bash
./launch-autospeed.sh
```

## Output

The output will be displayed in a new window that shows object detection and tracking of the input video.
