# VisionPilot/DomainSeg - OpenADKit Demo

Containerized DomainSeg Demo for VisionPilot.

## Prerequisites

- Download the [DomainSeg PyTorch model weights](https://github.com/autowarefoundation/autoware.privately-owned-vehicles/tree/main/Models#domainseg---roadwork-scene-segmentation) and place it in the `model-weights` directory with the name `domainseg.pth`.

    ```bash
    mkdir -p model-weights
    curl "https://drive.usercontent.google.com/download?id=1sYa2ltivJZEWMsTFZXAOaHK--Ovnadu2&confirm=xxx" -o model-weights/domainseg.pth
    ```

## Usage

```bash
./launch-domainseg.sh
```

## Output

"The output will be displayed in a new window that shows roadwork scene segmentation of the input image.
