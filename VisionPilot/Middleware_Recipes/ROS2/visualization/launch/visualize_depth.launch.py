import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    depth_topic_arg = DeclareLaunchArgument(
        'depth_topic',
        default_value='/auto3d/scene_3d/depth_map',
        description='Input depth topic to visualize'
    )
    
    output_topic_arg = DeclareLaunchArgument(
        'output_topic',
        default_value='/auto3d/scene_3d/viz',
        description='Output visualization topic'
    )

    # Depth visualization node
    visualize_depth_node = Node(
        package='visualization',
        executable='visualize_depth_node_exe',
        name='depth_visualizer',
        parameters=[{
            'depth_topic': LaunchConfiguration('depth_topic'),
            'output_topic': LaunchConfiguration('output_topic')
        }],
        output='screen'
    )

    return LaunchDescription([
        depth_topic_arg,
        output_topic_arg,
        visualize_depth_node
    ])