#include "run_model_node.hpp"
#include "../../common/include/onnx_runtime_backend.hpp"
#include "../../common/include/tensorrt_backend.hpp"
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace autoware_pov::vision
{

RunModelNode::RunModelNode(const rclcpp::NodeOptions & options)
: Node("run_model_node", options)
{
  // Get parameters
  const std::string model_path = this->declare_parameter<std::string>("model_path");
  const std::string backend_str = this->declare_parameter<std::string>("backend", "onnxruntime");
  const std::string precision = this->declare_parameter<std::string>("precision", "cpu");
  const int gpu_id = this->declare_parameter<int>("gpu_id", 0);
  const std::string input_topic = this->declare_parameter<std::string>("input_topic", "/sensors/camera/image_raw");
  output_topic_str_ = this->declare_parameter<std::string>("output_topic");
  model_type_ = this->declare_parameter<std::string>("model_type");
  measure_latency_ = this->declare_parameter<bool>("measure_latency", true);

  // Instantiate the backend
  if (backend_str == "onnxruntime") {
    backend_ = std::make_unique<OnnxRuntimeBackend>(model_path, precision, gpu_id);
  } else if (backend_str == "tensorrt") {
    backend_ = std::make_unique<TensorRTBackend>(model_path, precision, gpu_id);
  } else {
    RCLCPP_ERROR(this->get_logger(), "Unknown backend: %s", backend_str.c_str());
    throw std::invalid_argument("Unknown backend type.");
  }

  // Setup publisher and subscriber
  pub_ = image_transport::create_publisher(this, output_topic_str_);
  sub_ = image_transport::create_subscription(
    this, input_topic, std::bind(&RunModelNode::onImage, this, std::placeholders::_1), "raw",
    rmw_qos_profile_sensor_data);
  
  RCLCPP_INFO(this->get_logger(), "Model runner node configured for model type '%s'", model_type_.c_str());
  RCLCPP_INFO(this->get_logger(), "Input topic: %s", input_topic.c_str());
  RCLCPP_INFO(this->get_logger(), "Output topic: %s", output_topic_str_.c_str());
}

void RunModelNode::onImage(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  cv_bridge::CvImagePtr in_image_ptr;
  try {
    in_image_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (const cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  // --- Latency Watcher Start ---
  if (measure_latency_ && (++frame_count_ % LATENCY_SAMPLE_INTERVAL == 0)) {
    inference_start_time_ = std::chrono::steady_clock::now();
  }
  // -----------------------------

  if (!backend_->doInference(in_image_ptr->image)) {
    RCLCPP_WARN(this->get_logger(), "Inference failed.");
    return;
  }

  // --- Latency Watcher End & Report ---
  if (measure_latency_ && (frame_count_ % LATENCY_SAMPLE_INTERVAL == 0)) {
    auto inference_end_time = std::chrono::steady_clock::now();
    auto latency_ms =
      std::chrono::duration<double, std::milli>(inference_end_time - inference_start_time_)
        .count();
    RCLCPP_INFO(
      this->get_logger(), "Frame %zu: Inference Latency: %.2f ms (%.1f FPS)", frame_count_,
      latency_ms, 1000.0 / latency_ms);
  }
  // ------------------------------------

  // Get processed output from backend (like original architecture)
  cv::Mat processed_output;
  backend_->getRawOutput(processed_output, in_image_ptr->image.size(), model_type_);

  // Publish based on model type
  sensor_msgs::msg::Image::SharedPtr out_msg;
  if (model_type_ == "segmentation") {
    // Segmentation: backend returns CV_8UC1 mask (like original getRawMask/getDomainMask)
    out_msg = cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::MONO8, processed_output).toImageMsg();
  } else if (model_type_ == "depth") {
    // Depth: backend returns CV_32FC1 depth map
    out_msg = cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::TYPE_32FC1, processed_output).toImageMsg();
  } else {
    RCLCPP_WARN(this->get_logger(), "Unknown model type: %s", model_type_.c_str());
    return;
  }
  
  pub_.publish(out_msg);
}

}  // namespace autoware_pov::vision

RCLCPP_COMPONENTS_REGISTER_NODE(autoware_pov::vision::RunModelNode)