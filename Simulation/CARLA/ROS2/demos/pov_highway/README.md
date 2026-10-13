# VisionPilot closed-loop demo on PoV_Highway (CARLA 0.9.16)

VisionPilot drives the ego vehicle on a highway in CARLA, closed loop, from a single front
camera. CARLA publishes the camera natively over ROS 2; VisionPilot plans steering and
acceleration; two relays from `../../src` feed speed back and apply the commands.

What you see:

- **VisionPilot** window: camera view with lane/path overlay and the closest in-path object.
- **Occupancy** window: heuristic 3D bird's-eye view of detected cars (drag: rotate,
  right/middle drag: pan, `+`/`-`: zoom, `R`: reset).
- The ego (`hero`, Lincoln MKZ 2020) keeps its lane among 30 sedan NPCs at 60 +/- 10 km/h and
  slows behind slower traffic. With `--windowed`, the CARLA window shows the scene.

## Files

| File                        | Purpose                                                                                 |
| --------------------------- | --------------------------------------------------------------------------------------- |
| `setup.sh`                  | Checks prerequisites, builds the image, VisionPilot, relays, staged config. Idempotent. |
| `run_demo.sh`               | Starts CARLA, VisionPilot, the scenario and the relays; `--check-map` smoke test.       |
| `stop_demo.sh`              | Removes the demo containers and kills the CARLA server it started.                      |
| `params.env`                | All tunable conditions and paths (env vars override).                                   |
| `common.sh`                 | Helpers shared by the scripts (paths, container names, `docker run` wrapper).           |
| `pov_highway.py`            | Scenario: `check`, `traffic` (ego rig + NPCs), `kickstart` (rolling start).             |
| `make_ground_homography.py` | Analytic ground homography for the camera in `../../config/carla916.json`.              |
| `Dockerfile`                | CUDA 12.8 + Ubuntu 24.04 + ROS 2 Jazzy + ONNX Runtime 1.23.2 GPU dev image.             |

Everything generated goes to `$VP_DEMO_WORK` (default `./work/`, git-ignored). Nothing under
`VisionPilot/` is modified; the repo is mounted read-only into the containers.

## Prerequisites

- Linux x86_64 with an NVIDIA GPU (driver >= 570, CUDA 12.8 capable), Docker and
  [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/).
  Check: `docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi`.
- An X display (`echo $DISPLAY` is non-empty) and `xhost` (package `x11-xserver-utils`).
- `ss` (iproute2), `md5sum`, `tar`, ~25 GB free disk (CARLA ~20 GB, image ~8 GB, map 2.4 GB).
- Network access during `setup.sh` (apt, ROS apt repo, GitHub, gitlab.com for Eigen).
- No host Python is needed: all Python runs in the demo image with CARLA's cp312 wheel.

Environment used by every step (set once per shell; adjust paths):

```bash
export CARLA_ROOT=$HOME/CARLA_0.9.16_pov                       # CARLA working copy (step 1)
export MAP_TAR=$HOME/Downloads/20260701_PoV_Highway_0.9.16.tar.gz
cd <repo>/Simulation/CARLA/ROS2/demos/pov_highway
```

Optional: `VP_DEMO_WORK`, `VP_DEMO_IMAGE`, `VP_DEMO_PREFIX`, `VP_CUDA_CACHE`; see `params.env`.

## Step 0: get the map package

PoV_Highway is the Robotec.ai Unity/AWSIM highway environment, migrated to CARLA 0.9.16 by
community contributors ([autoware-projects#74](https://github.com/autowarefoundation/autoware-projects/issues/74),
[autoware.trucks#4](https://github.com/autowarefoundation/autoware.trucks/issues/4)).
It is **not** redistributed in this repository.

1. Open the "PoV_Highway map only" link posted by **@w-czerski** in
   [autoware.trucks#4](https://github.com/autowarefoundation/autoware.trucks/issues/4#issuecomment-4844812448):
   <https://drive.google.com/file/d/12UisDUYW9QRIP2JT7sisniGW9kZR-6oc/view>.
2. The file requires access: click **Request access** in Google Drive and, if needed, ask
   @w-czerski in a comment on that issue.
3. Download `20260701_PoV_Highway_0.9.16.tar.gz` (~2.4 GB) and verify it:

   ```bash
   md5sum "$MAP_TAR"     # expect 86735969c081c6d13ebe287380c01c59
   file "$MAP_TAR"       # expect: POSIX tar archive (GNU) -- not gzip despite the name
   ```

Use this original-map package only. Other packages that contain `BP_RoadManager` (from a
custom CARLA shipping build) crash stock CARLA at load with
`Could not find SuperStruct RoadManager`; `setup.sh` refuses such an install.

Never copy the package or its contents into this repository.

## Step 1: stock CARLA 0.9.16 working copy

Download `CARLA_0.9.16.tar.gz` from the
[CARLA 0.9.16 release](https://github.com/carla-simulator/carla/releases/tag/0.9.16) and extract
it into a dedicated directory (the map install modifies it; keep a pristine copy if you need
stock CARLA elsewhere, or `cp -a` an existing stock install):

```bash
mkdir -p "$CARLA_ROOT"
tar -xzf CARLA_0.9.16.tar.gz -C "$CARLA_ROOT"
```

Verify: `cat "$CARLA_ROOT/VERSION"` prints `0.9.16`, and
`ls "$CARLA_ROOT"/PythonAPI/carla/dist/` lists `carla-0.9.16-cp312-*.whl`.

## Step 2: install the map

Verified method: extract over the release root, skipping files that already exist. This adds
538 new files and leaves the stock Town assets untouched.

```bash
cd "$CARLA_ROOT" && tar --skip-old-files -xf "$MAP_TAR" && cd -
```

Verify:

```bash
ls "$CARLA_ROOT"/CarlaUE4/Content/Carla/Maps/PoV_Highway.umap \
   "$CARLA_ROOT"/CarlaUE4/Content/Carla/Maps/OpenDrive/PoV_Highway.xodr
```

Alternative (not verified): the standard CARLA route, `cp "$MAP_TAR" "$CARLA_ROOT/Import/"`
then `cd "$CARLA_ROOT" && ./ImportAssets.sh`. It is expected to work but overwrites 1,307
shared stock assets with the (newer) package copies.

## Step 3: build

```bash
./setup.sh
```

First run takes ~15-25 min (image build, then VisionPilot). It stages the carla cp312 wheel,
copies `VisionPilot/config` to `$VP_DEMO_WORK/config` and patches only `source.mode = ros2`,
`speed_limit`, `visualization_on` and `source.input_camera_topic`, generates the ground
homography `H.yaml` for the CARLA camera and the preprocess matrix `homography_C_matrix.yaml`
(via `VisionPilot/scripts/find_homography_C_matrix.py`), builds `carla_msgs` +
`carla_control_publisher` with colcon, and builds VisionPilot with
`-DENABLE_ROS2_INTERFACE=ON -DENABLE_OCCUPANCY=ON -DCMAKE_CXX_STANDARD_LIBRARIES=-lcppad_lib`.

Verify: the last line is `[pov_highway] setup complete: ...`, and

```bash
ls -l work/build/VisionPilot work/ros_ws/install/setup.bash work/config/H.yaml
grep -E '^(source\.mode|speed_limit)' work/config/vision_pilot.conf   # ros2, 25.0
```

## Step 4: smoke-test the map

```bash
./run_demo.sh --check-map
```

Starts CARLA offscreen, loads the map, prints its stats and stops CARLA (~1-2 min). Expect
`waypoints_2m=29608`, `spawn_points=2` and an `ego_start` near `x=-325.8, y=6673.8, yaw=-90`.

## Step 5: run the demo

```bash
./run_demo.sh                    # offscreen CARLA, NPCs 60 km/h, seed 7
./run_demo.sh --windowed         # also show the CARLA window
./run_demo.sh --npc-kmh 80 --seed 3
```

| Option                       | Env var             | Default | Meaning                                                           |
| ---------------------------- | ------------------- | ------- | ----------------------------------------------------------------- |
| `--npc-kmh N`                | `NPC_KMH`           | 60      | mean NPC speed (+/- 10 km/h per car)                              |
| `--seed N`                   | `SEED`              | 7       | seeds NPC models, colours, speeds and the Traffic Manager         |
| `--windowed` / `--offscreen` | `CARLA_WINDOWED`    | 0       | CARLA render window on `$DISPLAY`                                 |
|                              | `NPC_COUNT`         | 30      | NPCs, split over lanes -1/-2                                      |
|                              | `VP_SPEED_LIMIT_MS` | 25.0    | VisionPilot `speed_limit`, m/s (re-run `setup.sh` after changing) |
|                              | `KICKSTART_MS`      | 5.0     | rolling-start speed, m/s                                          |

Sequence: CARLA (`-nosound --ros2`, `FASTDDS_BUILTIN_TRANSPORTS=UDPv4`) -> `vp_pov_drive`
(VisionPilot, ~35 s model load, started early) -> `vp_pov_scenario` (loads PoV_Highway, spawns
NPCs and the ego rig from `../../config/carla916.json` at road 2, lane -2, s=505) ->
`vp_pov_speed` + `vp_pov_ctrl` relays -> rolling start once VisionPilot logs its first `plan:`.
The command returns after ~1-3 min with `[pov_highway] running. ...`.

Verify:

```bash
docker ps --filter name=vp_pov --format '{{.Names}}'   # drive, scenario, speed, ctrl
docker logs vp_pov_scenario | grep spawned              # spawned N npcs (N <= 30)
docker logs --tail 2 vp_pov_drive                       # plan: ... vel=+N.NN m/s, N > 0
```

What to look at: `cte` (lateral offset) stays within a few tenths of a metre, `tyre` near 0,
`cipo=true` with a plausible `dist` behind a lead car, and `accel` going negative when closing
in. `docker logs vp_pov_speed` shows `hero_actor found`.

## Step 6: stop

```bash
./stop_demo.sh
```

Verify: `docker ps -q --filter name=vp_pov` prints nothing and
`ss -ltnH 'sport = :2000'` prints nothing.

## Troubleshooting

| Symptom                                         | Cause / fix                                                                                                                                                                                                                                      |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `port 2000 is in use`                           | Another CARLA is running. Stop it (this demo only kills the server it started).                                                                                                                                                                  |
| `/vehicle/speed` stays 0, ego never moves       | Relays started before the map loaded and hold a stale world. `run_demo.sh` orders this; restart with `./run_demo.sh`.                                                                                                                            |
| Ego stays at 0, throttle and brake alternate    | Standing-start deadlock: `carla_control_publisher` targets `v + 2*a`, which stays ~0 from rest. `run_demo.sh` kick-starts the ego at `KICKSTART_MS`. Proper fix belongs in `carla_control_publisher` (e.g. a minimum target speed when `a > 0`). |
| `timed out ... VisionPilot's first plan`        | No camera frames: check `docker logs vp_pov_drive`, `$VP_DEMO_WORK/run/carla.log`; first CUDA start can be slow, raise `VP_PLAN_TIMEOUT`.                                                                                                        |
| VisionPilot window does not appear              | `DISPLAY` wrong or X access denied: `xhost +SI:localuser:$(id -un)` (or `xhost +local:`).                                                                                                                                                        |
| `carla client X != server Y` warning, segfaults | PythonAPI must be 0.9.16. Only the staged cp312 wheel from `$CARLA_ROOT` is used; a host env with carla 0.10.x is a mismatch.                                                                                                                    |
| `Could not find SuperStruct RoadManager`        | Wrong map package (Step 0). Re-create the CARLA copy and install the 20260701 package.                                                                                                                                                           |
| `PoV_Highway not installed`                     | Step 2 not done in this `$CARLA_ROOT`.                                                                                                                                                                                                           |
| Ego rear-ends a microlino, truck, van           | Those models are excluded on purpose (`EXCLUDED_MODELS` in `pov_highway.py`): VisionPilot mis-ranges them.                                                                                                                                       |
| CARLA keeps running after a crash               | `./stop_demo.sh`; it kills the server's whole session with SIGKILL (UE4 ignores SIGTERM).                                                                                                                                                        |

## Known limitations

- Observed (about 170 s runs, sedan-only NPCs): lateral offset < 0.4 m, steering ~0, no
  collisions, brakes to ~25 km/h behind 50-60 km/h traffic. Timing differs slightly between
  runs even with the same seed (asynchronous CARLA).
- The road is nearly straight: curve handling is untested. No oncoming traffic, no junctions.
- The map has no speed limits (CARLA's 30 km/h default), so NPC speeds are set as a Traffic
  Manager percentage: `(1 - target_kmh / 30) * 100`.
- Only 2 spawn points exist; positions use `get_waypoint_xodr(road, lane, s)`.
- Occupancy is a heuristic: 2D detections projected on the ground plane, not true 3D.
- `H.yaml` is analytic (pinhole camera from the rig JSON, flat ground, metres from the front
  bumper at 2.45 m for the MKZ 2020); it is not a calibration.

## For AI coding agents

- Set `CARLA_ROOT` and `DISPLAY`; work from this directory. Run, in order:
  `./setup.sh` -> `./run_demo.sh --check-map` -> `./run_demo.sh` -> observe -> `./stop_demo.sh`.
- Success: each script exits 0; `run_demo.sh` prints `[pov_highway] running.`;
  `docker logs --tail 2 vp_pov_drive` shows `plan:` lines with `vel` > 0 and growing over ~10 s.
- Diagnostics: `docker logs vp_pov_{drive,scenario,speed,ctrl}`, `$VP_DEMO_WORK/run/carla.log`.
- If your agent or shell runs commands in a sandbox that blocks the Docker socket, `docker`
  exits with empty output: run the scripts outside the sandbox, in the foreground.
- Never `pkill -f` broad patterns (they can match your own shell); use `./stop_demo.sh`.
- Do not modify `VisionPilot/`. Never commit the map package, `work/`, or anything from
  `$CARLA_ROOT`.
