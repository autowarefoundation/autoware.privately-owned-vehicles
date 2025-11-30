# Configuring CARLA for Vision Pilot Testing

## How to Run
After CARLA simulator server is up and runnning, run
```sh
# CARLA_0.10.0 with UE5
python3 ros_carla_config.py -f config/VisionPilot_carla10.json -v -a

# CARLA_0.9.16 with UE4
python3 ros_carla_config.py -f config/VisionPilot_carla9.json -v -a
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
- [waypoints_publisher](../ROS/src/waypoints_publisher/README.md)
- [odom_publisher](../ROS/src/odom_publisher/README.md)
<!-- - control_msg_converter: converts autoware controller output from `autoware_control_msgs/Control` to `ros_carla_msgs/CarlaEgoVehicleControl.msg` -->

To use these packages
```sh
# In this current directory
colcon build
source install/setup.bash

ros2 run waypoints_publisher pub_waypoints_node

# In another terminal (remember to source)
ros2 run odom_publisher pub_odom_node
```

## VisionPilot Testing Pipeline (Simulated EgoPath & EgoLanes)
### How to run
1. Build & source [VisionPilot ROS2 workspace](../../../VisionPilot/ROS2/)
2. Build & source this [workspace](../ROS2/)
3. Run [CARLA server with ROS2 enabled](../../README.md)
4. Wait for world to be loaded, then launch VisionPilot pipeline
   ```sh 
   # CARLA 0.9.16
   ros2 launch vision_pilot_bringup demo_carla9_launch.py    
   
   # CARLA 0.10.0
   ros2 launch vision_pilot_bringup demo_carla10_launch.py 
    ```
![](../../../Media/VisionPilot_demo.gif)CARLA 0.10.0