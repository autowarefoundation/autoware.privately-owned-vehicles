## Steering Controller

ROS2 C++ wrapper node responsible for computing the required steering angle to follow the desired path.

## Dependencies
- [SteeringController C++ lib](./../../../Control/Steering/SteeringController/README.md)

## Subscription
- /pathfinder/tracked_states (`Float32MultiArray`)
- /hero/odom (`nav_msgs/Odometry`)

## Publisher
- /vehicle/steering_cmd (`Float32`)

## Setup Instructions
1. Build [SteeringController C++ lib](./../../../Control/Steering/SteeringController/README.md)
2. Build ROS2 ws
   ```sh
   cd .../VisionPilot/ROS2/
   colcon build --packages-select steering_controller --symlink-install
3. Source & run
    ```sh
    source install/setup.bash
    
    # run as standalone node
    ros2 run steering_controller steering_controller_node
