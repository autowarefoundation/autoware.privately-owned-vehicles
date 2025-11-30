## Driving Stereo

#### Academic Paper: https://www.cvlibs.net/publications/Geiger2012CVPR.pdf

The [**Driving Stereo Dataset**](https://drivingstereo-dataset.github.io/) is large-scale stereo dataset which contains images covering a diverse set of driving scenarios, and is hundreds of times larger than the KITTI stereo dataset. High-quality labels of disparity are produced by a model-guided filtering strategy from multi-frame LiDAR points. Compared with other dataset, the deep-learning models trained on our DrivingStereo achieve higher generalization accuracy in real-world driving scenes. The dataset yields **a total of 174,436 training samples with associated ground truth data.**

### process_driving_stereo.py

**Arguments**
```bash
python3 process_driving_stereo.py -r <data directory> -s <save directory>

-r, --root : root data directory filepath
-s, --save : root save directory filepath
```