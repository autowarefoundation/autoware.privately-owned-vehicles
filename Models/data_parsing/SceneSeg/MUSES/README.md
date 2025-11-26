
## MUSES

#### Academic paper: https://arxiv.org/abs/2401.12761

MUSES, the MUlti-SEnsor Semantic perception dataset, designed for driving under increased uncertainty and adverse conditions. MUSES comprises a diverse collection of images, evenly distributed across different combinations of weather conditions (clear, fog, rain, and snow) and illuminations (day time/night time). Each image in the dataset is accompanied by high-quality 2D semantic image labels.

### process_muses.py
Colormap values for unified semantic classes created from the MUSES dataset to SceneSeg format are as follows:

| MUSES Semantic Class  | MUSES RGB Label | SceneSeg Semantic Class | SceneSeg RGB Label |
| -------- | ------- | ------- | ------- |
|Sky| ![#4682B4](https://via.placeholder.com/10/4682B4?text=+) rgb(70,130,180)| Sky | ![#3DB8FF](https://via.placeholder.com/10/3DB8FF?text=+) rgb(61, 184, 255)|
|Building|![#464646](https://via.placeholder.com/10/464646?text=+) rgb(70, 70, 70)| Background Objects | ![#3D5DFF](https://via.placeholder.com/10/3D5DFF?text=+) rgb(61, 93, 255)|
|Pole|![#999999](https://via.placeholder.com/10/999999?text=+) rgb(153, 153, 153)| Background Objects | ![#3D5DFF](https://via.placeholder.com/10/3D5DFF?text=+) rgb(61, 93, 255)|
|Traffic Light|![#FAAA1E](https://via.placeholder.com/10/FAAA1E?text=+) rgb(250, 170, 30)| Background Objects | ![#3D5DFF](https://via.placeholder.com/10/3D5DFF?text=+) rgb(61, 93, 255)|
|Traffic Sign|![#DCDC00](https://via.placeholder.com/10/DCDC00?text=+) rgb(220, 220, 0)| Background Objects | ![#3D5DFF](https://via.placeholder.com/10/3D5DFF?text=+) rgb(61, 93, 255)|
|Vegetation|![#6B8E23](https://via.placeholder.com/10/6B8E23?text=+) rgb(107, 142, 35)| Background Objects | ![#3D5DFF](https://via.placeholder.com/10/3D5DFF?text=+) rgb(61, 93, 255)|
|Terrain|![#98FB98](https://via.placeholder.com/10/98FB98?text=+) rgb(152, 251, 152)| Background Objects | ![#3D5DFF](https://via.placeholder.com/10/3D5DFF?text=+) rgb(61, 93, 255)|
|Unlabelled|![#000000](https://via.placeholder.com/10/000000?text=+) rgb(0,0,0)| Background Objects | ![#3D5DFF](https://via.placeholder.com/10/3D5DFF?text=+) rgb(61, 93, 255)|
|Person|![#DC143C](https://via.placeholder.com/10/DC143C?text=+) rgb(220, 20, 60)| Vulnerable Living | ![#FF3D3D](https://via.placeholder.com/10/FF3D3D?text=+) rgb(255, 61, 61)|
|Rider|![#FF0000](https://via.placeholder.com/10/FF0000?text=+) rgb(255, 0, 0)| Small Mobile Vehicle | ![#FFBE3D](https://via.placeholder.com/10/FFBE3D?text=+) rgb(255, 190, 61)|
|Motorcylce|![#0000E6](https://via.placeholder.com/10/0000E6?text=+) rgb(0, 0, 230)| Small Mobile Vehicle | ![#FFBE3D](https://via.placeholder.com/10/FFBE3D?text=+) rgb(255, 190, 61)|
|Bicycle|![#770B20](https://via.placeholder.com/10/770B20?text=+) rgb(119, 11, 32)| Small Mobile Vehicle | ![#FFBE3D](https://via.placeholder.com/10/FFBE3D?text=+) rgb(255, 190, 61)|
|Car|![#00008E](https://via.placeholder.com/10/00008E?text=+) rgb(0, 0, 142)| Large Mobile Vehicle | ![#FF743D](https://via.placeholder.com/10/FF743D?text=+) rgb(255, 116, 61) |
|Truck|![#000046](https://via.placeholder.com/10/000046?text=+) rgb(0, 0, 70)| Large Mobile Vehicle | ![#FF743D](https://via.placeholder.com/10/FF743D?text=+) rgb(255, 116, 61) |
|Bus|![#003C64](https://via.placeholder.com/10/003C64?text=+) rgb(0, 60, 100)| Large Mobile Vehicle | ![#FF743D](https://via.placeholder.com/10/FF743D?text=+) rgb(255, 116, 61) |
|Train|![#005064](https://via.placeholder.com/10/005064?text=+) rgb(0, 80, 100)| Large Mobile Vehicle | ![#FF743D](https://via.placeholder.com/10/FF743D?text=+) rgb(255, 116, 61) |
|Wall|![#66669C](https://via.placeholder.com/10/66669C?text=+) rgb(102, 102, 156)| Road Edge Delimiter | ![#D8FF3D](https://via.placeholder.com/10/D8FF3D?text=+) rgb(216, 255, 61)|
|Fence|![#BE9999](https://via.placeholder.com/10/BE9999?text=+) rgb(190, 153, 153)| Road Edge Delimiter | ![#D8FF3D](https://via.placeholder.com/10/D8FF3D?text=+) rgb(216, 255, 61)|
|Road| ![#804080](https://via.placeholder.com/10/804080?text=+) rgb(128, 64, 128)| Road | ![#00FFDC](https://via.placeholder.com/10/00FFDC?text=+) rgb(0, 255, 220) |




