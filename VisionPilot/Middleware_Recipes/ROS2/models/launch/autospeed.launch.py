import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    models_pkg_dir = get_package_share_directory('models')
    
    # Model path - relative to workspace
    default_model_path = os.path.join(models_pkg_dir, '..', '..', '..', '..', 
                                      'data', 'models', 'AutoSpeed_n.pt')
    
    # Declare launch arguments
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value=default_model_path,
        description='Path to AutoSpeed model (.pt or .onnx)'
    )
    precision_arg = DeclareLaunchArgument(
        'precision',
        default_value='fp32',
        description='Precision: fp32 or fp16'
    )
    benchmark_arg = DeclareLaunchArgument(
        'benchmark',
        default_value='false',
        description='Enable benchmark/latency measurements'
    )
    
    # AutoSpeed detection node
    autospeed_node = Node(
        package='models',
        executable='autospeed_node_exe',
        name='autospeed_detection',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'precision': LaunchConfiguration('precision'),
            'gpu_id': 0,
            'benchmark': LaunchConfiguration('benchmark'),
            'conf_threshold': 0.6,
            'iou_threshold': 0.45,
            'input_topic': '/sensors/video/image_raw',
            'output_topic': '/autospeed/detections'
        }],
        output='screen'
    )

    return LaunchDescription([
        model_path_arg,
        precision_arg,
        benchmark_arg,
        autospeed_node
    ])

