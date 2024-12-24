## GTAV

#### Academic Paper: https://arxiv.org/abs/2306.01704

The [**GTAV Stereo Dataset**](https://github.com/ostadabbas/Temporal-controlled-Frame-Swap-GTAV-TeFS-) consists of time-synchronized stereo images captured from the famous Grand Theft Auto V role-play simulation game. The authors of the dataset also provide ground-truth disparity maps (equivalent to max depth of up to 600m) for both sets of stereo camera images and release a sample set of data logs. For the purpose of training SuperDepth, we focused our attention on road driving scenes and did not utilize the off-road driving sequences of the dataset. The 'City05_AirportTraffic' data sample had a broken link. Data samples were temporally downsampled to prevent over-fitting, resulting in **a total of 1,230 training samples with associated ground truth data.**

### process_gtav.py

**Arguments**
```bash
python3 process_gtav.py -r <data directory> -s <save directory>

-r, --root : root data directory filepath
-s, --save : root save directory filepath
```
**Dataset Parameters**
- Stereo camera baseline distance: 0.54m
- Camera height above road surface: 2.1m
- Max Height in height map: 7m
- Min Height in height map: -5m