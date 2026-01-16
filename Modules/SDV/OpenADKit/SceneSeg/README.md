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

After the container is running, you can access the visualization by opening the following URL in your browser:

<http://localhost:6080/vnc.html?resize=scale&autoconnect=true&password=visualizer>

The output shows semantic segmentation of the scene of the input video in real-time.
