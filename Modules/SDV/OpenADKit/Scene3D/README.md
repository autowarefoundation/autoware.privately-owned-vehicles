# Scene3D - Open AD Kit Demo

Containerized Scene3D Demo, monocular depth estimation.

## Prerequisites

- Download the [Scene3D PyTorch model weights](https://github.com/autowarefoundation/autoware.privately-owned-vehicles/tree/main/Models#scene3d---monocular-depth-estimation) and place it in the `model-weights` directory with the name `scene3d.pth`.

    ```bash
    mkdir -p model-weights
    curl "https://drive.usercontent.google.com/download?id=1MrKhfEkR0fVJt-SdZEc0QwjwVDumPf7B&confirm=xxx" -o model-weights/scene3d.pth
    ```

## Usage

```bash
./launch-scene3d.sh
```

## Output

The output will be displayed in a new window that shows monocular depth estimation of the input image.
