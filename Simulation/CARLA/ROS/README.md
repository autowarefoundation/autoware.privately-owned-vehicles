# Configuring CARLA for Vision Pilot Testing

## How to Run
After CARLA simulator server is up and runnning, run
```sh
python3 ros_carla_config.py -f config/VisionPilot.json -v -a
```
This script spawns the ego vehicle and sensors, enables the ROS2 interface and the spectator view will follow the vehicle in the simulator. The arg `-v` enables verbose output and `-a` enables CARLA's built-in Traffic Manager autopilot (off by-default), which is suitable for testing Perception models.

### RVIZ2 Visualization
```sh
ros2 run rviz2 rviz2 -d config/VisionPilot.rviz
```

## Sensor Configurations
To add/remove sensors or change sensor attributes such as Range, FOV and mounting pose, modify or create a copy of `config/VisionPilot.json`. The current configuration is for testing SAE L3 single lane Vision Pilot.

![](../../../Media/Roadmap.jpg)

List of sensors available in CARLA 0.10.0: https://carla-ue5.readthedocs.io/en/latest/ref_sensors/

## Control Command
For testing controllers or the full perception-control pipeline, run `ros_carla_config.py` without `-a` autopilot on and publish [`ros_carla_msgs/CarlaEgoVehicleControl.msg`](https://carla-ue5.readthedocs.io/en/latest/ros2_native/#:~:text=ego/vehicle_control_cmd.-,CarlaEgoVehicleControl.msg,-To%20send%20control) to `/carla/hero/vehicle_control_cmd` topic. Install the package separately from https://github.com/carla-simulator/ros-carla-msgs/tree/master.

## CARLA-Autoware Custom Interfaces
- waypoints_publisher : creates `autoware_planning_msgs/Trajectory`
<!-- - control_msg_converter: converts autoware controller output from `autoware_control_msgs/Control` to `ros_carla_msgs/CarlaEgoVehicleControl.msg` -->