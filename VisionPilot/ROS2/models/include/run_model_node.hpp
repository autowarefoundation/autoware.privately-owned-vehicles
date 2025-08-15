#ifndef RUN_MODEL_NODE_HPP_
#define RUN_MODEL_NODE_HPP_

#include "../../common/include/inference_backend_base.hpp"
#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <memory>
#include <chrono>

namespace autoware_pov::vision
{

class RunModelNode : public rclcpp::Node
{
public:
  explicit RunModelNode(const rclcpp::NodeOptions & options);

private:
  void onImage(const sensor_msgs::msg::Image::ConstSharedPtr msg);

  // Parameters
  std::string model_type_;
  std::string output_topic_str_;
  
  // Backend
  std::unique_ptr<InferenceBackend> backend_;

  // ROS
  image_transport::Subscriber sub_;
  image_transport::Publisher pub_;
  
  // Latency monitoring (like original AUTOSEG)
  static constexpr size_t LATENCY_SAMPLE_INTERVAL = 100; // Log every 100 frames
  size_t frame_count_ = 0;
  bool measure_latency_ = false;
  std::chrono::steady_clock::time_point inference_start_time_;
};

}  // namespace autoware_pov::vision

#endif  // RUN_MODEL_NODE_HPP_