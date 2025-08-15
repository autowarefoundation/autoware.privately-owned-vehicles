import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml
from launch.substitutions import PythonExpression

def generate_launch_description():
    # Get the package share directory for models
    models_pkg_dir = get_package_share_directory('models')

    # --- Declare Launch Arguments ---
    video_path_arg = DeclareLaunchArgument(
        'video_path',
        description='Path to the video file to be used by the sensor node.'
    )
    pipeline_arg = DeclareLaunchArgument(
        'pipeline',
        default_value='scene_seg',
        description='Which pipeline to run: scene_seg, domain_seg, or scene_3d.'
    )

    # --- Sensor Node ---
    # This node is common to all pipelines
    video_publisher_node = Node(
        package='sensors',
        executable='video_publisher_node_exe',
        name='video_publisher',
        parameters=[{
            'video_path': LaunchConfiguration('video_path'),
            'output_topic': '/sensors/video/image_raw',
            'loop': True
        }],
        output='screen'
    )

    # --- Configuration Files (YAML controls everything) ---
    autoseg_param_file = os.path.join(models_pkg_dir, 'config', 'autoseg.yaml')
    scene3d_param_file = os.path.join(models_pkg_dir, 'config', 'auto3d.yaml')


    # --- Pipeline Nodes ---
    
    # Scene Segmentation Nodes
    scene_seg_model_node = Node(
        package='models', executable='models_node_exe', name='scene_seg_model',
        parameters=[autoseg_param_file], output='screen',
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('pipeline'), "' == 'scene_seg'"]))
    )
    scene_seg_viz_node = Node(
        package='visualization', executable='visualize_masks_node_exe', name='scene_seg_viz',
        parameters=[{
            'image_topic': '/sensors/video/image_raw',
            'mask_topic': '/autoseg/scene_seg/mask',
            'viz_type': 'scene',
            'output_topic': '/autoseg/scene_seg/viz'
        }],
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('pipeline'), "' == 'scene_seg'"]))
    )

    # Domain Segmentation Nodes
    domain_seg_model_node = Node(
        package='models', executable='models_node_exe', name='domain_seg_model',
        parameters=[autoseg_param_file], output='screen',
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('pipeline'), "' == 'domain_seg'"]))
    )
    domain_seg_viz_node = Node(
        package='visualization', executable='visualize_masks_node_exe', name='domain_seg_viz',
        parameters=[{
            'image_topic': '/sensors/video/image_raw',
            'mask_topic': '/autoseg/domain_seg/mask',
            'viz_type': 'domain',
            'output_topic': '/autoseg/domain_seg/viz'
        }],
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('pipeline'), "' == 'domain_seg'"]))
    )

    # Scene 3D (Depth Estimation) Nodes
    scene3d_model_node = Node(
        package='models', executable='models_node_exe', name='scene3d_model',
        parameters=[scene3d_param_file], output='screen',
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('pipeline'), "' == 'scene_3d'"]))
    )
    scene3d_viz_node = Node(
        package='visualization', executable='visualize_depth_node_exe', name='scene3d_viz',
        parameters=[{
            'depth_topic': '/auto3d/scene_3d/depth_map',
            'output_topic': '/auto3d/scene_3d/viz'
        }],
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('pipeline'), "' == 'scene_3d'"]))
    )

    return LaunchDescription([
        video_path_arg,
        pipeline_arg,
        video_publisher_node,
        scene_seg_model_node,
        scene_seg_viz_node,
        domain_seg_model_node,
        domain_seg_viz_node,
        scene3d_model_node,
        scene3d_viz_node
    ])