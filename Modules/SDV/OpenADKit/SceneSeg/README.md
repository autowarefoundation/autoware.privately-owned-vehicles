# SceneSeg - Open AD Kit Demo

Containerized SceneSeg Demo, semantic segmentation of the scene.

## Prerequisites

- Download the [SceneSeg PyTorch model weights](https://github.com/autowarefoundation/autoware.privately-owned-vehicles/tree/main/Models#sceneseg---semantic-segmentation-of-the-scene) and place it in the `model-weights` directory with the name `sceneseg.pth`.

    ```bash
    mkdir -p model-weights
    curl "https://drive.usercontent.google.com/download?id=1vCZMdtd8ZbSyHn1LCZrbNKMK7PQvJHxj&confirm=xxx" -o model-weights/sceneseg.pth
    ```

## Usage

```bash
./launch-sceneseg.sh
```

## Output

The output will be displayed in a new window that shows semantic segmentation of the scene of the input video.
