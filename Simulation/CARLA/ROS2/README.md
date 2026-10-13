# VisionPilot ⇄ CARLA (0.9.16) — ROS 2

In order to connect VisionPilot to CARLA for closed loop simulation and to not introduce carla_bridge dependency, two
nodes are required, one to publish vehicle speed and one to publish a command message so CARLA can drive the vehicle.

`carla_vehicle_speed_publisher` package publishes the vehicle speed to `/vehicle/speed` topic, and
`carla_control_publisher` publish control messages to `/carla/hero/vehicle_control_cmd`.

## Demos

- [`demos/pov_highway`](demos/pov_highway/README.md): VisionPilot closed loop on the PoV_Highway map
  with Traffic Manager traffic (stock CARLA 0.9.16 + map package, one-command start/stop).
