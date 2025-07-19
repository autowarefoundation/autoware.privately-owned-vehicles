#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <image_transport/image_transport.hpp>
#include <chrono>

#include "scene3d.hpp"

namespace autoware_pov::AutoSeg::Scene3D
{

class Scene3DNode : public rclcpp::Node
{
public:
  explicit Scene3DNode(const rclcpp::NodeOptions & node_options);

private:
  void onConnect();
  void onImage(const sensor_msgs::msg::Image::ConstSharedPtr msg);

  std::unique_ptr<Scene3D> scene3d_;
  
  // Publishers
  image_transport::Publisher depth_pub_;
  image_transport::Publisher color_depth_pub_;
  
  // Subscriber 
  image_transport::Subscriber image_sub_;
  
  // Timer for connection management
  rclcpp::TimerBase::SharedPtr timer_;
  
  // Latency monitoring
  static constexpr size_t LATENCY_SAMPLE_INTERVAL = 200;
  size_t frame_count_ = 0;
  bool measure_latency_ = false;
  std::chrono::steady_clock::time_point inference_start_time_;
};

}  // namespace autoware_pov::AutoSeg::Scene3D 