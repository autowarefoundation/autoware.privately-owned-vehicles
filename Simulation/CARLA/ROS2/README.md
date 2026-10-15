# VisionPilot ⇄ CARLA (0.9.16) — ROS 2

In order to connect VisionPilot to CARLA for closed loop simulation and to not introduce carla_bridge dependency, two
nodes are required, one to publish vehicle speed and one to publish a command message so CARLA can drive the vehicle.

`carla_vehicle_speed_publisher` package publishes the vehicle speed to `/vehicle/speed` topic, and
`carla_control_publisher` publish control messages to `/carla/hero/vehicle_control_cmd`.

## PoV Highway map (optional)

Closed loop on the PoV Highway map with 30 Traffic Manager cars (stock CARLA 0.9.16).

1. Request the map package `20260701_PoV_Highway_0.9.16.tar.gz` from its owner (see
   `autowarefoundation/autoware.trucks#4`, "PoV_Highway map only"). It is not redistributable.
2. Install it on a copy of CARLA 0.9.16:
   `cp -r CARLA_0.9.16 CARLA_pov && cd CARLA_pov && tar --skip-old-files -xf /path/to/20260701_PoV_Highway_0.9.16.tar.gz`
3. Start CARLA with `./CarlaUE4.sh --ros2` and wait until it has fully loaded.
4. Launch the bridge with the PoV rig (the `carla` Python wheel of 0.9.16 must be importable):

```bash
ros2 launch carla_bridge_bringup carla_bridge.launch.py map:=PoV_Highway \
  rig_file:=$PWD/config/carla916_pov_highway.json
```

5. Once the camera topic publishes, start VisionPilot last, with `source.mode = ros2`,
   `source.input_camera_topic = /carla/hero/main_cam/image` and `speed_limit = 25.0`
   (the default 33.3 m/s rear-ended slower traffic in testing). The CARLA camera needs its own
   homography: copy `config/H_carla916.yaml` to `VisionPilot/config/H.yaml` before building.

The rig spawns the ego on road 2 / lane -2 / s=505 and seeds the traffic (`seed`). The map has no
speed limit (CARLA default 30 km/h), so `speed_diff_percent: -100` makes the NPCs drive at about
60 km/h.
