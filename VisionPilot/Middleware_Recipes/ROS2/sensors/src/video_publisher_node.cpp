#include "video_publisher_node.hpp"
#include <rclcpp_components/register_node_macro.hpp>

namespace autoware_pov::sensors
{

VideoPublisherNode::VideoPublisherNode(const rclcpp::NodeOptions & options)
: Node("video_publisher", options),
  frame_count_(0)
{
  // Declare parameters
  video_path_ = this->declare_parameter<std::string>("video_path", "");
  output_topic_ = this->declare_parameter<std::string>("output_topic", "/sensors/video/image_raw");
  fps_ = this->declare_parameter<double>("fps", 60.0);
  loop_ = this->declare_parameter<bool>("loop", true);

  // Validate video path
  if (video_path_.empty()) {
    RCLCPP_ERROR(this->get_logger(), "video_path parameter is required!");
    return;
  }

  // Open video file
  cap_.open(video_path_);
  if (!cap_.isOpened()) {
    RCLCPP_ERROR(this->get_logger(), "Failed to open video file: %s", video_path_.c_str());
    return;
  }

  // Create publisher
  image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>(output_topic_, 10);

  // Create timer
  auto timer_period = std::chrono::milliseconds(static_cast<int>(1000.0 / fps_));
  timer_ = this->create_wall_timer(timer_period, std::bind(&VideoPublisherNode::publishFrame, this));

  RCLCPP_INFO(this->get_logger(), "Video publisher started. Publishing to: %s", output_topic_.c_str());
}

void VideoPublisherNode::publishFrame()
{
  cv::Mat frame;
  if (!cap_.read(frame)) {
    if (loop_) {
      // Restart video
      cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
      frame_count_ = 0;
      if (!cap_.read(frame)) {
        RCLCPP_ERROR(this->get_logger(), "Failed to restart video");
        return;
      }
    } else {
      RCLCPP_INFO(this->get_logger(), "Video finished");
      return;
    }
  }

  // Convert to ROS message
  auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
  msg->header.stamp = this->get_clock()->now();
  msg->header.frame_id = "camera_frame";

  // Publish
  image_publisher_->publish(*msg);
  frame_count_++;
}

}  // namespace autoware_pov::sensors

RCLCPP_COMPONENTS_REGISTER_NODE(autoware_pov::sensors::VideoPublisherNode)