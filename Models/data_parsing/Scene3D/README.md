# Scene3D Relative Depth Dataset

In order to create the relative depth dataset used to train Scene3D, DepthAnythingV2-Large model was utilized to generate pseudo-labels for a variety of street scene images capturing variations in time of day, weather conditions, camera mounting height, camera mounting angle, camera lens type (fisheye to zoom). Please see examples below of DepthAnythingV2 on real-world data and its comparions with DepthAnythingV1:

![DepthAnythingV2 Examples](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/assets/teaser.png)

## Street-level scenes
Specifically, images from the following street scenes datasets were proceesed to create pseudo labels, resulting in a total dataset size of 488,535 images with associated pseudo labels. The KITTI, DDAD and DrivingStereo datasets were temporally downsampled to prevent model over-fitting:

- [BDD100K](https://www.kaggle.com/datasets/marquis03/bdd100k)
- [Mapillary Vistas](https://www.mapillary.com/dataset/vistas)
- [ROADWork](https://www.cs.cmu.edu/~ILIM/roadwork_dataset/)
- [Comma10K](https://github.com/commaai/comma10k)
- [KITTI](https://www.cvlibs.net/datasets/kitti/raw_data.php)
- [DDAD](https://github.com/TRI-ML/DDAD#dataset-details)
- [Driving Stereo](https://drivingstereo-dataset.github.io/)
- [Indian Driving Dataset](https://idd.insaan.iiit.ac.in/)
- [Zenesact Open Dataset](https://zod.zenseact.com/frames/)
- [Mapillary Planet Scale dataset - Images only](https://www.mapillary.com/dataset/depth)


## DepthAnythingV2-Large Pseudo Labels

To create pseudo-labels, please clone the [DepthAnythingV2 repository](https://github.com/DepthAnything/Depth-Anything-V2/tree/main) and download the [VIT-Large Model weights](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true) to be able to run inference. Copy-paste the script below into the main DepthAnythingV2 repo folder, save it as create_pseudo_label.py and simply run as shown below (please make sure that the output directory has a /depth and /image folder for saving the pseudo labels and processed image):

### Example Usage
```bash
python3 create_pseudo_label.py --img-path /path/to/input/image/folder --outdir /path/to/where/data/is/saved
```

### create_pseudo_label.py
```python
import argparse
import cv2
import pathlib
import numpy as np
import torch

from depth_anything_v2.dpt import DepthAnythingV2


if __name__ == '__main__':

    # Argument parsing
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    parser.add_argument('--img-path', dest='img-path')
    parser.add_argument('--outdir', dest='save-path')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--input-size', type=int, default=518)
    args = parser.parse_args()
    
    # Inference device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Model configuration - we use vitl for pseudo labels
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    # Load Model from checkpoint and set to inference device
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load \
        (f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    # Data Filepaths
    images_filepath = args.img-path
    image_save_filepath = args.save-path + '/images'
    depth_filepath = args.save-path + '/depth'

    # Reading images
    images = sorted([f for f in pathlib.Path(images_filepath).glob("*.jpg")])
    print(len(images), ' samples found')
    
    # Main loop to process images
    for index in range(0, len(images)):

        print('processing image ', index, ' of ', len(images)-1)

        # Reading image using OpenCV and getting dimensions
        image = cv2.imread(str(images[index]))
        height, width, channels = image.shape

        # Resizing image if it is very large
        if(width > 800 and width <= 1300):
            image = cv2.resize(image, (0,0), fx=0.8, fy=0.8)

        if(width > 1300 and width <= 1800):
            image = cv2.resize(image, (0,0), fx=0.5, fy=0.5) 

        if(width > 1800 and width <= 2600):
            image = cv2.resize(image, (0,0), fx=0.4, fy=0.4) 

        if(width > 2600):
            image = cv2.resize(image, (0,0), fx=0.25, fy=0.25) 

        # Run inference
        depth = depth_anything.infer_image(image)

        # Save raw depth-map
        depth_save_path = depth_filepath + str(index) + '.npy'
        np.save(depth_save_path, depth)

        # Save associated image
        image_save_path = image_save_filepath + str(index) + '.jpg'
        cv2.imwrite(image_save_path, image)

    print('Processing complete')
```

## Scene3D Metric Depth Dataset
To train the Scene3D metric depth network, a custom dataset was created using a number of open-source, publicly available datasets which capture scene depth through LIDAR scanners. Raw depth maps were interpolated to fill in holes and create a densified depth ground-truth. Validity maps were created to account for areas of missing data which would not be penalized during network training.

The Scene3D Metric Depth Dataset comprises RGB images with associated ground truth depth maps and validity masks indicating pixels where valid depth measurements are present.

![Scene3D_Metric_Data](../../Diagrams/Scene3D_Metric_Data.jpg)

The datasets used to train SuperDepth include:

- [Argoverse](https://www.argoverse.org/av1.html#stereo-link)
- [KITTI](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction)
- [DDAD](https://github.com/TRI-ML/DDAD#dataset-details)
- [DrivingStereo](https://drivingstereo-dataset.github.io/)

## common
Contains helper class to perform LIDAR depth map interpolation and densification