## EgoLanes
EgoLanes is a neural network that processes raw image frames and performs real-time semantic segmentation of driving lanes in the image. It produces a three class segmentation output for the ego-left lane, the ego-right lane and all other lanes. It outputs lanes at 1/4 resolution of the input image size allowing for quick inference on low power embedded hardware. EgoLanes was trained with data from a variety of real-world datasets including TuSimple, OpenLane, CurveLanes, Jiqing, and ONCE3D Lane.

### Loss Function:

- Lane-level binary cross-entropy loss: This loss penalized the model in its individual class level predictions
- Edge preservation loss: This loss ensured the model was able to predict lane boundaries accurately