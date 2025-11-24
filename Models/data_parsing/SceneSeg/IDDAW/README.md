
## IDDAW

#### Academic paper: https://ar5iv.labs.arxiv.org/html/2311.14459

The IDDAW (Indian Driving Dataset Adverse Weather) provides images captured on Indian roads in a variety of weather and lighting conditions across highways, rural roads and complex urban scenes. The ground-truth semantic labels are stored in a JSON file which contains polygon object contours according to semantic class labels. There are a total of 30 different semantic classes provided in the dataset. 

### process_bdd100k.py
Colormap values for unified semantic classes created from the IDDAW dataset to SceneSeg format are as follows:

| IDDAW Semantic Class Label in JSON | SceneSeg Semantic Class | SceneSeg RGB Label |
| -------- | ------- | ------- |
|`sky`| Sky | ![#3DB8FF](https://via.placeholder.com/10/3DB8FF?text=+) rgb(61, 184, 255)|
|`billboard`| Background Objects | ![#3D5DFF](https://via.placeholder.com/10/3D5DFF?text=+) rgb(61, 93, 255)|
|`traffic sign`| Background Objects | ![#3D5DFF](https://via.placeholder.com/10/3D5DFF?text=+) rgb(61, 93, 255)|
|`traffic light`| Background Objects | ![#3D5DFF](https://via.placeholder.com/10/3D5DFF?text=+) rgb(61, 93, 255)|
|`pole`| Background Objects | ![#3D5DFF](https://via.placeholder.com/10/3D5DFF?text=+) rgb(61, 93, 255)|
|`obs-str-bar-fallback`|Background Objects | ![#3D5DFF](https://via.placeholder.com/10/3D5DFF?text=+) rgb(61, 93, 255)|
|`building`| Background Objects | ![#3D5DFF](https://via.placeholder.com/10/3D5DFF?text=+) rgb(61, 93, 255)|
|`bridge`| Background Objects | ![#3D5DFF](https://via.placeholder.com/10/3D5DFF?text=+) rgb(61, 93, 255)|
|`vegetation`| Background Objects | ![#3D5DFF](https://via.placeholder.com/10/3D5DFF?text=+) rgb(61, 93, 255)|
|`fallback background`| Background Objects | ![#3D5DFF](https://via.placeholder.com/10/3D5DFF?text=+) rgb(61, 93, 255)|
|`parking`| Background Objects | ![#3D5DFF](https://via.placeholder.com/10/3D5DFF?text=+) rgb(61, 93, 255)|
|`drivable-fallback`| Background Objects | ![#3D5DFF](https://via.placeholder.com/10/3D5DFF?text=+) rgb(61, 93, 255)|
|`sidewalk`| Background Objects | ![#3D5DFF](https://via.placeholder.com/10/3D5DFF?text=+) rgb(61, 93, 255)|
|`non-drivable fallback`| Background Objects | ![#3D5DFF](https://via.placeholder.com/10/3D5DFF?text=+) rgb(61, 93, 255)|
|`person`|Vulnerable Living | ![#FF3D3D](https://via.placeholder.com/10/FF3D3D?text=+) rgb(255, 61, 61)|
|`animal`|Vulnerable Living | ![#FF3D3D](https://via.placeholder.com/10/FF3D3D?text=+) rgb(255, 61, 61)|
|`rider`|Small Mobile Vehicle | ![#FFBE3D](https://via.placeholder.com/10/FFBE3D?text=+) rgb(255, 190, 61)|
|`motorcycle`| Small Mobile Vehicle | ![#FFBE3D](https://via.placeholder.com/10/FFBE3D?text=+) rgb(255, 190, 61)|
|`bicycle`| Small Mobile Vehicle | ![#FFBE3D](https://via.placeholder.com/10/FFBE3D?text=+) rgb(255, 190, 61)|
|`autorickshaw`| Large Mobile Vehicle | ![#FF743D](https://via.placeholder.com/10/FF743D?text=+) rgb(255, 116, 61) |
|`car`|Large Mobile Vehicle | ![#FF743D](https://via.placeholder.com/10/FF743D?text=+) rgb(255, 116, 61) |
|`truck`| Large Mobile Vehicle | ![#FF743D](https://via.placeholder.com/10/FF743D?text=+) rgb(255, 116, 61) |
|`bus`|Large Mobile Vehicle | ![#FF743D](https://via.placeholder.com/10/FF743D?text=+) rgb(255, 116, 61) |
|`caravan`|Large Mobile Vehicle | ![#FF743D](https://via.placeholder.com/10/FF743D?text=+) rgb(255, 116, 61) |
|`vehicle fallback`|Large Mobile Vehicle | ![#FF743D](https://via.placeholder.com/10/FF743D?text=+) rgb(255, 116, 61) |
|`curb`|Road Edge Delimiter | ![#D8FF3D](https://via.placeholder.com/10/D8FF3D?text=+) rgb(216, 255, 61)|
|`wall`| Road Edge Delimiter | ![#D8FF3D](https://via.placeholder.com/10/D8FF3D?text=+) rgb(216, 255, 61)|
|`fence`|Road Edge Delimiter | ![#D8FF3D](https://via.placeholder.com/10/D8FF3D?text=+) rgb(216, 255, 61)|
|`guard rail`| Road Edge Delimiter | ![#D8FF3D](https://via.placeholder.com/10/D8FF3D?text=+) rgb(216, 255, 61)|
|`road`| Road | ![#00FFDC](https://via.placeholder.com/10/00FFDC?text=+) rgb(0, 255, 220) |