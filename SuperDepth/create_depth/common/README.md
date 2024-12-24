
## lidar_depth_fill.py
In order to train the SuperDepth network, we require ground truth depth maps. In real world datasets, ground druth depth maps are acquired by LIDAR scanners, resulting in sparse virtual depth maps created by proejction of 3D lidar points onto in image, where, in most cases, depth data is only available for 5% of pixels. This script creates a fully filled in lidar depth map based on the method [In Defense of Classical Image Processing: Fast Depth Completion on the CPU](https://arxiv.org/abs/1802.00036)

![Lidar Depth Fill](../../../Diagrams/Lidar_Depth_Fill.jpg)

## stereo_sparse_supervision.py
As part of the SuperDepth network, sparse stereo supervision is used to help guide the network in capturing the tre scene scale. In order to create sparse stereo supervision, the OpenCV block-matching algorithm is utilized to find sparse stereo correspondences between the left and right stereo image pair. Median blurring is utilized to remove speckle noise and the resulting data is transformed to a height map.

## height_map.py
Helper class to convert depth map into height map based on camera parameters. The origin of the height map is set to the ground point right in front of the camera. This results in a height map that is road-relative, making it agnostic to camera mounting height.

## depth_boundaries.py
Depth estimation networks struggle to accurately predict depth at foreground-background object boundaries where there is a step-change in the depth value. These depth boundary pixels are calculated through a first-order discrete derivative filter kernel in both the x and y directions over the depth map. The absolute sum of x and y derviates is then thresholded to identify depth boundary pixels.

## depth_sparse_supervision.py
For those datasets which do not provide stereo images, a type of pseudo stereo supervision is performed. This is done by first calculating salient features in the RGB image where there is a texture change, through a first order discrete derivate filter in the Green colour channel. Those pixels with colour texture information above a threshold are then used as a mask to extract corresponding depth values. This builds on the assumption that sparse stereo matching algorithms leverage regions with strong texture information for correspondence matching and disparity calculation.

