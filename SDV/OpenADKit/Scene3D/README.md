# VisionPilot/Scene3D - OpenADKit Demo

Containerized Scene3D Demo for VisionPilot.

## Prerequisites

- Docker

- Download the [Scene3D PyTorch model weights](https://github.com/autowarefoundation/autoware.privately-owned-vehicles/tree/main/Models#scene3d---monocular-depth-estimation) and place it in the `model-weights` directory with the name `scene3d.pth`.

    ```bash
    mkdir -p model-weights
    curl "https://drive.usercontent.google.com/download?id=19gMPt_1z4eujo4jm5XKuH-8eafh-wJC6&confirm=xxx" -o model-weights/scene3d.pth
    ```

- Download the [Test image](https://drive.google.com/file/d/100rOuKXAFqaW7iZ5KHlamRtaSHoRVJaq/view?usp=drive_link) and place it in the `test` directory with the name `image.jpg`.

    ```bash
    mkdir -p test
    curl "https://drive.usercontent.google.com/download?id=100rOuKXAFqaW7iZ5KHlamRtaSHoRVJaq&confirm=xxx" -o test/image.jpg
    ```

## Usage

```bash
./launch-scene3d.sh
```

## Output

The output will be displayed in a new window that shows monocular depth estimation of the input image.
