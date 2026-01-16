# EgoLanes - Open AD Kit Demo

Containerized EgoLanes Demo, semantic segmentation of driving lanes.

## Prerequisites

- Download the [EgoLanes PyTorch model weights](https://github.com/autowarefoundation/autoware.privately-owned-vehicles/tree/main/Models#egolanes---semantic-segmentation-of-driving-lanes) and place it in the `model-weights` directory with the name `egolanes.pth`.

    ```bash
    mkdir -p model-weights
    curl "https://drive.usercontent.google.com/download?id=1Njo9EEc2tdU1ffo8AUQ9mjwuQ9CzSRPX&confirm=xxx" -o model-weights/egolanes.pth
    ```

## Usage

```bash
./launch-egolanes.sh
```

## Output

The output will be displayed in a new window that shows semantic segmentation of the driving lanes of the input video.
