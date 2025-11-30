import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    mask_topic_arg = DeclareLaunchArgument(
        'mask_topic',
        default_value='/autoseg/scene_seg/mask',
        description='Input mask topic to visualize'
    )
    
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/sensors/video/image_raw',
        description='Input image topic for blending'
    )
    
    output_topic_arg = DeclareLaunchArgument(
        'output_topic',
        default_value='/autoseg/scene_seg/viz',
        description='Output visualization topic'
    )

    # Scene segmentation visualization node
    visualize_scene_seg_node = Node(
        package='visualization',
        executable='visualize_masks_node_exe',
        name='scene_seg_visualizer',
        parameters=[{
            'mask_topic': LaunchConfiguration('mask_topic'),
            'image_topic': LaunchConfiguration('image_topic'),
            'output_topic': LaunchConfiguration('output_topic'),
            'viz_type': 'scene'
        }],
        output='screen'
    )

    return LaunchDescription([
        mask_topic_arg,
        image_topic_arg,
        output_topic_arg,
        visualize_scene_seg_node
    ])