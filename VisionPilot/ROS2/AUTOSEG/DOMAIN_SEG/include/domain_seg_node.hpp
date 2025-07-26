#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <image_transport/image_transport.hpp>
#include <chrono>

#include "domain_seg.hpp"

namespace autoware_pov::AutoSeg::DomainSeg
{

class DomainSegNode : public rclcpp::Node
{
public:
  explicit DomainSegNode(const rclcpp::NodeOptions & node_options);

private:
  void onConnect();
  void onImage(const sensor_msgs::msg::Image::ConstSharedPtr msg);

  std::unique_ptr<DomainSeg> domain_seg_;
  
  // Publishers
  image_transport::Publisher domain_pub_;
  image_transport::Publisher color_domain_pub_;
  image_transport::Publisher blended_pub_;
  
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

}  // namespace autoware_pov::AutoSeg::DomainSeg 