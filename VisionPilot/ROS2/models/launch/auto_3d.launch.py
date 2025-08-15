import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare the launch arguments
    model_name_arg = DeclareLaunchArgument(
        'model_name',
        default_value='scene3d_model',
        description='The name of the model to launch'
    )
    param_file_arg = DeclareLaunchArgument(
        'param_file',
        default_value='VisionPilot/ROS2/models/config/auto3d.yaml',
        description='Path to the YAML file with model parameters'
    )

    # Node definition - YAML controls everything
    run_model_node = Node(
        package='models',
        executable='models_node_exe',
        name=LaunchConfiguration('model_name'),
        parameters=[LaunchConfiguration('param_file')],
        output='screen'
    )

    return LaunchDescription([
        model_name_arg,
        param_file_arg,
        run_model_node
    ])