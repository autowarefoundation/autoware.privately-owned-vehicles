## augmentations.py
Image augmentation is an essential part of training visual neural networks. It helps harden the network to various noise effects during training and helps to reduce data over-fitting. The [albumentations library](https://albumentations.ai/) is used to create different image augmentations effects simulation weather, debris, lens effects, noise, colour shifting, and mixing of image patches

![Augmentations Network Diagram](../../Diagrams/Augmentations.jpg)

## check_data.py
Script to perform a sanity check on data for processing, ensuring that data is read and that number of ground-truth samples match the number of training images

## load_data_scene_seg.py
Helper class for the [SceneSeg Neural network](https://github.com/autowarefoundation/autoware.privately-owned-vehicles/tree/main/SceneSeg) to load multiple datasets, separate data into training and validation splits and extract a Region of Interest (ROI) from images

## benchmark.py
Script to print model layers, number of parameters, and measure inference speed of model at either FP32 or FP16 precision