## MUAD

#### Academic Paper: https://arxiv.org/abs/2203.01437

[**MUAD (Multiple Uncertainties for Autonomous Driving)**](https://muad-dataset.github.io/), is a synthetic dataset for autonomous driving with multiple uncertainty types and tasks. It contains 3420 images in the train set and 492 in the validation set, covering day and night conditions, with 2/3 being day images and 1/3 night images. Alongside a set monocular camera images, the per-pixel depth is also provided wtih a max range of up to 400m in 'exr' format. **This results in a total of 3,912 training samples with associated ground truth data.**

### process_muad.py

**Arguments**
```bash
python3 process_muad.py -r <data directory> -s <save directory>

-r, --root : root data directory filepath
-s, --save : root save directory filepath
```
**Dataset Parameters**
- Camera height above road surface: 1.35m
- Max Height in height map: 7m
- Min Height in height map: -2m