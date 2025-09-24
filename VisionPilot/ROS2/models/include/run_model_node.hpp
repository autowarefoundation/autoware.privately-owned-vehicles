#ifndef RUN_MODEL_NODE_HPP_
#define RUN_MODEL_NODE_HPP_

#include "../../common/include/fps_timer.hpp"
#include "../../common/include/inference_backend_base.hpp"
#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <memory>
#include <chrono>

#define BENCHMARK_OUTPUT_FREQUENCY 100

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
  std::string gpu_backend_; // "cuda" or "hip"
  
  // Backend
  std::unique_ptr<InferenceBackend> backend_;

  // ROS
  image_transport::Subscriber sub_;
  image_transport::Publisher pub_;

  // Benchmark: Timer
  FpsTimer timer_;
};

}  // namespace autoware_pov::vision

#endif  // RUN_MODEL_NODE_HPP_