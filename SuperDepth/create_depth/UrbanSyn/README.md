## UrbanSyn

#### Academic Paper: https://arxiv.org/abs/1911.02620

The [**UrbanSyn Dataset**](https://www.urbansyn.org/) is an open synthetic dataset featuring photorealistic driving scenes. It contains ground-truth annotations for semantic segmentation, scene depth, panoptic instance segmentation, and 2-D bounding boxes, providing more than 7.5k synthetic annotated images. It was born to address the synth-to-real domain gap, contributing to unprecedented synthetic-only baselines used by domain adaptation (DA) methods. After accounting for samples in the test set for which data was not available, there are **a total of 6,732 training samples with associated ground truth data.**

### process_urban_syn.py

**Arguments**
```bash
python3 process_urban_syn.py -r <data directory> -s <save directory>

-r, --root : root data directory filepath
-s, --save : root save directory filepath
```
**Dataset Parameters**
- Camera height above road surface: 1.2m
- Max Height in height map: 7m
- Min Height in height map: -2m