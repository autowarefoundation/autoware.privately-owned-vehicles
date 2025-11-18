import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Get absolute path to autoseg.yaml
    workspace_root = os.getcwd()
    default_param_file = os.path.join(workspace_root, 'models/config/autoseg.yaml')

    # Declare the launch arguments
    scene_seg_name_arg = DeclareLaunchArgument(
        'scene_seg_name',
        default_value='scene_seg_model',
        description='The name of the model to launch (scene_seg_model or domain_seg_model)'
    )

    domain_seg_name_arg = DeclareLaunchArgument(
        'domain_seg_name',
        default_value='domain_seg_model',
        description='The name of the model to launch (scene_seg_model or domain_seg_model)'
    )

    autoseg_param_file_arg = DeclareLaunchArgument(
        'param_file',
        default_value=default_param_file,
        description='Path to the YAML file with model parameters'
    )

    auto3d_param_file_arg = DeclareLaunchArgument(
        'param_file',
        default_value='models/config/auto3d.yaml',
        description='Path to the YAML file with model parameters'
    )

    # Node definition - YAML controls everything
    scene_seg_node = Node(
        package='models',
        executable='models_node_exe',
        name=LaunchConfiguration('scene_seg_name'),
        parameters=[LaunchConfiguration('param_file')],
        output='screen'
    )

    # Node definition - YAML controls everything
    domain_seg_node = Node(
        package='models',
        executable='models_node_exe',
        name=LaunchConfiguration('domain_seg_name'),
        parameters=[LaunchConfiguration('param_file')],
        output='screen'
    )

    # Node definition - YAML controls everything
    scene3d_node = Node(
        package='models',
        executable='models_node_exe',
        name='scene3d_model',
        parameters=[LaunchConfiguration('param_file')],
        output='screen'
    )

    return LaunchDescription([
        scene_seg_name_arg,
        domain_seg_name_arg,
        autoseg_param_file_arg,
        auto3d_param_file_arg,
        scene_seg_node,
        domain_seg_node,
        scene3d_node
    ])