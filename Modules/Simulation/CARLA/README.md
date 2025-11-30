# ROS2
Ensure that CARLA server is running with `--ros2` args, then open a new terminal and run
```sh
ros2 topic echo /clock
```
If CARLA is fully rendered and running, timestamps messages should be logged. If not, jump to `Troubleshooting` section below.

There should not be much topics available right now as the CARLA world is not configured yet and the ego vehicle has not been spawned along with the sensors, follow [HERE](../CARLA/ROS/README.md) on how to do so.

## Demo
Here is how to run the autopilot example CARLA provided
```sh
cd .../Carla-0.10.0-Linux-Shipping/PythonAPI/examples/ros2
python3 ros2_native.py -f stack.json
```

### RVIZ2 Visualization
Run the script provided from the CARLA installation to run another docker image
```sh
cd .../Carla-0.10.0-Linux-Shipping/PythonAPI/examples/ros2
./run_rviz.sh
```
or locally 
```sh
cd .../Carla-0.10.0-Linux-Shipping/PythonAPI/examples/ros2
ros2 run rviz2 rviz2 -d rviz/ros2_native.rviz
```

![](../../Media/carla_ros.gif)

## Troubleshooting
In a local terminal outside of the docker images, if the ROS2 topics are available but no messages are getting through. Add the following to `~/.bashrc` locally, source and try again.
```sh
# Modify file path as needed
export FASTRTPS_DEFAULT_PROFILES_FILE=~/Carla-0.10.0-Linux-Shipping/PythonAPI/examples/ros2/config/fastrtps-profile.xml 
```