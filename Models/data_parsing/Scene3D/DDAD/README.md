## Dense Depth for Autonomous Driving (DDAD)

#### Academic Paper: https://arxiv.org/abs/1905.02693

The [**DDAD Dataset**](https://github.com/TRI-ML/DDAD#dataset-details) is a new autonomous driving benchmark from TRI (Toyota Research Institute) for long range (up to 250m) and dense depth estimation in challenging and diverse urban conditions. It contains monocular videos and accurate ground-truth depth (across a full 360 degree field of view) generated from high-density LiDARs mounted on a fleet of self-driving cars operating in a cross-continental setting. DDAD contains scenes from urban settings in the United States (San Francisco, Bay Area, Cambridge, Detroit, Ann Arbor) and Japan (Tokyo, Odaiba). For the purpose of training SuperDepth, we only utilise data from LIDAR projected into the Front facing and Rear facing vehicle cameras - **yileding a total of 33,200 training data samples**. In order to perform the lidar-to-camera projection, you need to use the [Toyota Data Governance Policy Library](https://github.com/TRI-ML/dgp) which requires Docker. There may be some challenges in setting up Docker on computers that run Ubuntu using Windows WSL2 - in case of any such issues, please refer to this [troubleshooting guide](https://docs.google.com/document/d/1YQnBEANAvRQvGuAgNc4d6kBXQe1h8iEmBCdfs1NIu_I/edit?usp=sharing)

### process_ddad.py

**Arguments**
```bash
python3 process_ddad.py -d <ddad json filepathy> -s <save directory>

-d, --data : path to ddad.json file
-s, --save : root save directory filepath
```