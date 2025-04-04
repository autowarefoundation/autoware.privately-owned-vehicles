## Scene3D
Depth estimation is an essential technology for safe operation of self-driving cars, especially in challenging edge case scenarios. By sensing depth, self-driving cars are able to detect important objects in the scene irrespective of their appearance. Scene3D is able process monocular camera images to produce high resolution depth maps with sharp object boundaries, visible on the leaves of trees, thin structures such as poles, and on the edges of foreground objects - helping self-driving cars understand the dynamic driving scene in real-time. Scene3D enables important downstream perception tasks such as foreground obstacle detection, and is robust to changes in object appearance, size, shape and type, addressing 'long-tail' edge case scenarios. The current release of Scene3D estimates per-pixel relative depth, indicating which objects are nearer vs further away from the camera. Scene3D is part of the [AutoSeg Foundation Model](https://github.com/autowarefoundation/autoware.privately-owned-vehicles/tree/main/AutoSeg) which forms the core of the vision-pipeline of the [Autoware Autonomous Highway Pilot System](https://github.com/autowarefoundation/autoware.privately-owned-vehicles/tree/main)

![Scene3D GIF](../Media/Scene3D_GIF.gif) ![Scene3D GIF 2](../Media/Scene3D_GIF_2.gif)

## Watch the explainer video
Please click the video link to play - [***Video link***](https://drive.google.com/file/d/19E57_ECVF3ImMGY8TNmg7dqixH1ej8MB/view?usp=drive_link)

## Demo, Training, Inference, Visualization
Please see the [*Models*](https://github.com/autowarefoundation/autoware.privately-owned-vehicles/tree/main/Models) folder to access the pre-trained network weights for Scene3D as well as scripts for network training, inference and visualization of network predictions.

## Performance Results
Scene3D was trained on a diverse dataset comprised of multiple open-source datasets, including Mapillary BDD100K, Mapillary Vistas, ROADWork, Comma10K, KITTI, DDAD, Driving Stereo, Indian Driving Dataset, Zenesact Open Dataset and the Mapillary Planet Scale Dataset. RGB images from these datasets were processed using DepthAnythingV2 VIT-Large model to create a ground truth relative depth map pseudo labels - resulting in 488,535 total samples, of which 24,426 samples were set aside for validation.

During training, Scene3D estimates a relative depth map which is compared with the ground truth pseudo labels from DepthAnything to calculate the network loss. We utilize 2 loss functions, a scale invariant loss and an edge loss:

### Scale Invariant Loss
Multiple scale and scale shift invariant losses were tried, however, we found that the best results were achieved by applying min-max scaling to the predicted and ground truth depth pseudo label and then calcuating a robust mean absolute error (ignoring the top 10% of erroneous pixels as they were likely caused by erros in the ground truth pseudo labels)

### Edge Preservation Loss
In order to preserve sharp edges in the depth map, an edge preservation loss was applied which applies a 3x3 derivative kernel in x and y directions to calculate the edge gradients in the prediction and ground truth depth pseudo label and calculates the mean absolute error. To ensure smoothness and for regularization, this edge preservation loss was applied at multiple image scales (full-size, 1/2 , 1/4, 1/8). The final edge preservation loss was calculated as an average between the individual edge losses at multiple scales.

### Total Loss
The total loss is set to a weighted sum of the Scale Invariant Loss and Edge Preservation Loss to ensure both losses are of similar magnitude during training.

### Training Details
The network was trained in 2 stages, during the first stage the batch size was set to 24 and the learning rate was set to 1e-4. The best network from stage 1 (as measured on validation dataset results) was finetuned in stage 2 with a batch size of 3 and a learning rate of 1.25e-5

### Validation Set Performance - Loss
| Total Loss | Scale Invariant Loss | Edge Preservation Loss |
|------------|----------------------|------------------------|
| **0.08408** | 0.07004 | 0.01404 | 

### Inference Speed
Inference speed tests were performed on a laptop equipped with an RTX3060 Mobile Gaming GPU, and an AMD Ryzen 7 5800H CPU. 

#### FP32 Precision
At FP32 precision, Scene3D achieved 15.1 Frames Per Second inference speed

#### FP16 Precision
At FP16 precision, Scene3D achieved 23.9 Frames Per Second inference speed