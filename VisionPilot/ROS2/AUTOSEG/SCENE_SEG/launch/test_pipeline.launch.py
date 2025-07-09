
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_dir = get_package_share_directory("autoware_pov_scene_seg")

    # --- Declare Launch Arguments for the whole test pipeline ---
    declare_model_path_arg = DeclareLaunchArgument(
        "model_path",
        description="Path to the ONNX model file."
    )
    
    declare_video_path_arg = DeclareLaunchArgument(
        "video_path",
        description="Path to the input video file for the publisher."
    )

    declare_precision_arg = DeclareLaunchArgument(
        "precision",
        default_value="cpu",
        description="The execution provider to use for inference (e.g., 'cpu', 'cuda')."
    )

    # --- Video Publisher Node ---
    video_publisher_node = Node(
        package="autoware_pov_scene_seg",
        executable="video_publisher.py",
        name="video_publisher_node",
        output="screen",
        arguments=[
            LaunchConfiguration("video_path"), # video_path
            "/in/image",                       # topic_name
            "10.0"                             # frame_rate
        ],
    )

    # --- Include the Segmentation Node Launch File ---
    # This reuses the launch file we already created
    segmentation_node_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_dir, "launch", "scene_seg.launch.py"])
        ),
        launch_arguments={
            "model_path": LaunchConfiguration("model_path"),
            "precision": LaunchConfiguration("precision"),
            # Pass the correct topic from the publisher to the segmentation node
            "input_image_topic": "/in/image",
        }.items(),
    )

    return LaunchDescription([
        declare_model_path_arg,
        declare_video_path_arg,
        declare_precision_arg,
        video_publisher_node,
        segmentation_node_launch
    ]) 