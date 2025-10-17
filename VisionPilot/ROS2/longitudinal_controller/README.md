## Longitudinal Controller

ROS2 C++ node responsible for computing the required throttle and brake to reach and maintain the desired longitudinal speed.

## Dependencies
- [PI Controller C++ lib](./../../../Control/Longitudinal/PI_Controller/README.md)

## Subscription
- /odom (`nav_msgs/Odometry`)

## Publisher
- /vehicle/throttle_cmd (`Float32`)
## Steering Controller

## Setup Instructions
1. Build [PI Controller C++ lib](./../../../Control/Longitudinal/PI_Controller/README.md)
2. Build ROS2 ws
   ```sh
   cd .../VisionPilot/ROS2/
   colcon build --packages-select longitudinal_controller --symlink-install
3. Source & run
    ```sh
    source install/setup.bash
    
    # run as standalone node
    ros2 run longitudinal_controller longitudinal_controller_node
