## comma10K

#### Blog post: https://blog.comma.ai/crowdsourced-segnet-you-can-help/

The comma10K dataset comprises 10,000 PNGs of real driving sscenes captured from the comma fleet, semantically labeled by the public. The dataset contains images from both the road-facing cameras (wide angle, main) and internal-facing driver monitoring camera - however, only images from the road-facing cameras were included.

### process_comma10k.py
Colormap values for unified semantic classes created from the comma10K dataset to SceneSeg format are as follows:

| comma10K Semantic Class  | comma10K RGB Label | SceneSeg Semantic Class | SceneSeg RGB Label |
| -------- | ------- | ------- | ------- |
|Sky|Unavailable - created offline| Sky | ![#3DB8FF](https://via.placeholder.com/10/3DB8FF?text=+) rgb(61, 184, 255)|
|Undrivable|![#808060](https://via.placeholder.com/10/808060?text=+) rgb(128, 128, 96)| Background Objects | ![#3D5DFF](https://via.placeholder.com/10/3D5DFF?text=+) rgb(61, 93, 255)|
|Ego Vehicle|![#cc00ff](https://via.placeholder.com/10/cc00ff?text=+) rgb(128, 128, 96)| Background Objects | ![#3D5DFF](https://via.placeholder.com/10/3D5DFF?text=+) rgb(61, 93, 255)|
|Movable|![#00ff66](https://via.placeholder.com/10/00ff66?text=+) rgb(0, 255, 102)| Foreground Objects |![#FF1C91](https://via.placeholder.com/10/FF1C91?text=+) rgb(255, 28, 145) |
|Lane Markings| ![#ff0000](https://via.placeholder.com/10/ff0000?text=+) rgb(255, 0, 0)| Road | ![#00FFDC](https://via.placeholder.com/10/00FFDC?text=+) rgb(0, 255, 220) |
|Road| ![#402020](https://via.placeholder.com/10/402020?text=+) rgb(64, 32, 32)| Road | ![#00FFDC](https://via.placeholder.com/10/00FFDC?text=+) rgb(0, 255, 220) |