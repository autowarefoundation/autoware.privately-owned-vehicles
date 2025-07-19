from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_dir = get_package_share_directory("autoware_pov_scene3d")

    # --- Declare Launch Arguments for the whole test pipeline ---
    declare_model_path_arg = DeclareLaunchArgument(
        "model_path",
        description="Path to the ONNX model file for depth estimation."
    )
    
    declare_video_path_arg = DeclareLaunchArgument(
        "video_path",
        description="Path to the input video file for the publisher."
    )

    declare_backend_arg = DeclareLaunchArgument(
        "backend",
        default_value="tensorrt",
        description="The backend to use for inference (e.g., 'onnxruntime', 'tensorrt')."
    )

    declare_precision_arg = DeclareLaunchArgument(
        "precision",
        default_value="fp16",
        description="The execution provider/precision to use for inference (e.g., 'cpu', 'cuda', 'fp32', 'fp16')."
    )

    declare_gpu_id_arg = DeclareLaunchArgument(
        "gpu_id",
        default_value="0",
        description="The GPU ID to use for inference."
    )

    declare_measure_latency_arg = DeclareLaunchArgument(
        "measure_latency",
        default_value="true",
        description="Whether to measure latency."
    )

    declare_frame_rate_arg = DeclareLaunchArgument(
        "frame_rate",
        default_value="60.0",
        description="Frame rate for video publishing (FPS)."
    )

    # --- Video Publisher Node ---
    video_publisher_node = Node(
        package="autoware_pov_scene3d",
        executable="video_publisher.py",
        name="video_publisher_node",
        output="screen",
        arguments=[
            LaunchConfiguration("video_path"),     # video_path
            "/in/image",                          # topic_name
            LaunchConfiguration("frame_rate")     # frame_rate
        ],
    )

    # --- Include the Scene3D Node Launch File ---
    # This reuses the launch file we already created
    scene3d_node_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_dir, "launch", "scene3d.launch.py"])
        ),
        launch_arguments={
            "model_path": LaunchConfiguration("model_path"),
            "backend": LaunchConfiguration("backend"),
            "precision": LaunchConfiguration("precision"),
            "gpu_id": LaunchConfiguration("gpu_id"),
            "measure_latency": LaunchConfiguration("measure_latency"),
            # Pass the correct topic from the publisher to the depth estimation node
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
        scene3d_node_launch
    ]) 