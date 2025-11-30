## KITTI

#### Academic Paper: https://www.cvlibs.net/publications/Geiger2012CVPR.pdf

The [**KITTI Depth Prediction Dataset**](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) consists of accumulated scans from an HDL-64 LIDAR scanner, projected onto a pair of RGB and Grayscale camera images, arranged in a stereo configuration, comprising a total of 93,000 depth maps (up to a max depth of 120m). Data is collected by driving on the roads of Karlsruhe in Germany in Urban, Highway, and Campus environments during day-time under fair weather conditions. The KITTI benchmark is one of the first autonomous vehicle open datasets and benchmarks for various perception related tasks. For the purpose of training SuperDepth, we focused our attention on road driving scenes in Urban and Highway settings and did not utilize the 'Campus' and 'Person' driving sequences of the dataset, resulting in **a total of 43,240 training samples with associated ground truth data.**

### process_kitti.py

**Arguments**
```bash
python3 process_kitti.py -r <data directory> -s <save directory>

-r, --root : root data directory filepath
-s, --save : root save directory filepath
```