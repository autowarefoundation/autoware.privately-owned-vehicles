import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    pkg_dir = get_package_share_directory("autoware_pov_domain_seg")
    
    # --- Declare Launch Arguments ---
    declare_model_path_arg = DeclareLaunchArgument(
        "model_path",
        default_value=os.path.join(pkg_dir, "data", "domain_seg_model.onnx"),
        description="Path to the ONNX model file for domain segmentation."
    )
    
    declare_video_path_arg = DeclareLaunchArgument(
        "video_path",
        default_value=os.path.join(pkg_dir, "data", "test_video.mp4"),
        description="Path to the test video file."
    )
    
    declare_backend_arg = DeclareLaunchArgument(
        "backend",
        default_value="tensorrt",
        description="The backend to use for inference (e.g., 'onnxruntime', 'tensorrt')."
    )
    
    declare_precision_arg = DeclareLaunchArgument(
        "precision",
        default_value="fp32",
        description="The execution provider/precision to use for inference (e.g., 'cpu', 'cuda', 'fp32', 'fp16')."
    )

    declare_gpu_id_arg = DeclareLaunchArgument(
        "gpu_id",
        default_value="0",
        description="The GPU device ID to use for inference."
    )

    declare_measure_latency_arg = DeclareLaunchArgument(
        "measure_latency",
        default_value="true",
        description="Whether to measure and log inference latency."
    )

    declare_frame_rate_arg = DeclareLaunchArgument(
        "frame_rate",
        default_value="60.0",
        description="Frame rate for video playback."
    )

    # --- Video Publisher Node ---
    video_publisher_node = Node(
        package="autoware_pov_domain_seg",
        executable="video_publisher.py",
        name="video_publisher_node",
        arguments=[
            LaunchConfiguration("video_path"),
            "/in/image",
            LaunchConfiguration("frame_rate")
        ],
        output="screen"
    )

    # --- Domain Segmentation Node Launch ---
    # Use the dedicated domain_seg.launch.py file
    segmentation_node_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                get_package_share_directory("autoware_pov_domain_seg"),
                "launch",
                "domain_seg.launch.py"
            ])
        ]),
        launch_arguments={
            "model_path": LaunchConfiguration("model_path"),
            "backend": LaunchConfiguration("backend"),
            "precision": LaunchConfiguration("precision"),
            "gpu_id": LaunchConfiguration("gpu_id"),
            "measure_latency": LaunchConfiguration("measure_latency"),
            "input_image_topic": "/in/image",
        }.items(),
    )

    return LaunchDescription([
        declare_model_path_arg,
        declare_video_path_arg,
        declare_backend_arg,
        declare_precision_arg,
        declare_gpu_id_arg,
        declare_measure_latency_arg,
        declare_frame_rate_arg,
        video_publisher_node,
        segmentation_node_launch
    ]) 