#ifndef VIDEO_PUBLISHER_NODE_HPP_
#define VIDEO_PUBLISHER_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <string>

namespace autoware_pov::sensors
{

class VideoPublisherNode : public rclcpp::Node
{
public:
  explicit VideoPublisherNode(const rclcpp::NodeOptions & options);

private:
  void publishFrame();

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  cv::VideoCapture cap_;
  std::string video_path_;
  std::string output_topic_;
  double fps_;
  int frame_count_;
  bool loop_;
};

}  // namespace autoware_pov::sensors

#endif  // VIDEO_PUBLISHER_NODE_HPP_