## Argoverse 1.0

#### Academic Paper: https://arxiv.org/abs/1911.02620

The data in [**Argoverse 1.0**](https://www.argoverse.org/index.html) comes from a subset of the area in which Argo AI’s self-driving test vehicles are operating in Miami and Pittsburgh — two US cities with distinct urban driving challenges and local driving habits. It includes recordings of vehicle sensor data, or "log segments," across different seasons, weather conditions, and times of day to provide a broad range of real-world driving scenarios. For the purpose of training SuperDepth, we focus on the [**Stereo Vision dataset**](https://www.argoverse.org/av1.html#stereo-link) within Argoverse 1.0. The Stereo Vision dataset consists of rectified stereo images and ground truth disparity maps for 74 out of the 113 Argoverse 1.0  Tracking Sequences. The stereo images are (2056 x 2464 px) and sampled at 5 Hz. The dataset contains a total of 6,624 stereo pairs with ground truth depth (up to 200m range), although ground truth depth for the 15-sequence test set is not provided. Furthermore, a temporal downsampling is applied to the raw Stereo Vision data to create an effective image sampling rate of 1 Hz. This is important to ensure that the network does not over-fit to images which look similar to each other. **This results in a total of 1,106 training samples with associated ground truth data.**

### process_argoverse.py

**Arguments**
```bash
python3 process_argoverse.py -r <data directory> -s <save directory>

-r, --root : root data directory filepath
-s, --save : root save directory filepath
```
**Dataset Parameters**
- Stereo camera baseline distance: 0.2986m
- Camera height above road surface: 1.67m
- Max Height in height map: 7m
- Min Height in height map: -2m