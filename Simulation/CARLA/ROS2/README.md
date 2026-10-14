# VisionPilot ⇄ CARLA (0.9.16) — ROS 2

In order to connect VisionPilot to CARLA for closed loop simulation and to not introduce carla_bridge dependency, two
nodes are required, one to publish vehicle speed and one to publish a command message so CARLA can drive the vehicle.

`carla_vehicle_speed_publisher` package publishes the vehicle speed to `/vehicle/speed` topic, and
`carla_control_publisher` publish control messages to `/carla/hero/vehicle_control_cmd`.

## PoV Highway map (optional)

Runs the same closed loop on the PoV Highway map with 30 Traffic Manager cars (stock CARLA 0.9.16).

1. Request the map package `20260701_PoV_Highway_0.9.16.tar.gz` from its owner (see
   `autowarefoundation/autoware.trucks#4`, "PoV_Highway map only"). It is not redistributable.
2. Install it on a copy of CARLA 0.9.16:
   `cp -r CARLA_0.9.16 CARLA_pov && cd CARLA_pov && tar --skip-old-files -xf /path/to/20260701_PoV_Highway_0.9.16.tar.gz`
3. Start CARLA with `./CarlaUE4.sh --ros2`, then launch the bridge with the PoV rig:

```bash
ros2 launch carla_bridge_bringup carla_bridge.launch.py map:=PoV_Highway \
  rig_file:=$PWD/config/carla916_pov_highway.json
```

`config/carla916_pov_highway.json` spawns the ego on road 2 / lane -2 / s=505, seeds the traffic
(`seed`) and lists sedan-only NPCs. The map has no speed limit (CARLA default 30 km/h), so the NPC
`speed_diff_percent` of -100 makes them drive at about 60 km/h.
