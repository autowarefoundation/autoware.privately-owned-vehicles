#include "run_model_node.hpp"
#include "../../common/include/onnx_runtime_backend.hpp"
#include "../../common/include/tensorrt_backend.hpp"

#ifdef CUDA_FOUND
#include "../../common/include/cuda_visualization_kernels.hpp"
#endif
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

  // Get tensor data from backend
  const float* tensor_data = backend_->getRawTensorData();
  std::vector<int64_t> tensor_shape = backend_->getTensorShape();
  
  if (tensor_shape.size() != 4) {
    RCLCPP_ERROR(this->get_logger(), "Invalid tensor shape");
    return;
  }
  
  // Model-type specific processing
  if (model_type_ == "depth") {
    // Depth estimation: output raw depth values (CV_32FC1)
    int height = static_cast<int>(tensor_shape[2]);
    int width = static_cast<int>(tensor_shape[3]);
    
    // Create depth map from tensor data (single channel float)
    cv::Mat depth_map(height, width, CV_32FC1, const_cast<float*>(tensor_data));
    
    // Resize depth map to original image size (use LINEAR for depth)
    cv::Mat resized_depth;
    cv::resize(depth_map, resized_depth, in_image_ptr->image.size(), 0, 0, cv::INTER_LINEAR);
    
    // Publish depth map as CV_32FC1
    sensor_msgs::msg::Image::SharedPtr out_msg = 
      cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::TYPE_32FC1, resized_depth).toImageMsg();
    pub_.publish(out_msg);
    
  } else if (model_type_ == "segmentation") {
    // Segmentation: create binary masks
    cv::Mat mask;
    
#ifdef CUDA_FOUND
    // Try CUDA acceleration first
    bool cuda_success = CudaVisualizationKernels::createMaskFromTensorCUDA(
      tensor_data, tensor_shape, mask
    );
    
    if (!cuda_success) {
#endif
      // CPU fallback: create mask from tensor
      int height = static_cast<int>(tensor_shape[2]);
      int width = static_cast<int>(tensor_shape[3]);
      int channels = static_cast<int>(tensor_shape[1]);
      
      mask = cv::Mat(height, width, CV_8UC1);
      
      if (channels > 1) {
        // Multi-class segmentation: argmax across channels (NCHW format)
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            float max_score = -1e9f;
            uint8_t best_class = 0;
            for (int c = 0; c < channels; ++c) {
              // NCHW format: tensor_data[batch=0][channel=c][height=h][width=w]
              float score = tensor_data[c * height * width + h * width + w];
              if (score > max_score) {
                max_score = score;
                best_class = static_cast<uint8_t>(c);
              }
            }
            // Convert class IDs for scene segmentation: Class 1 -> 255, others -> 0
            mask.at<uint8_t>(h, w) = (best_class == 1) ? 255 : 0;
          }
        }
      } else {
        // Single channel: threshold for binary segmentation
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            float value = tensor_data[h * width + w];
            mask.at<uint8_t>(h, w) = (value > 0.0f) ? 255 : 0;
          }
        }
      }
#ifdef CUDA_FOUND
    }
#endif
    
    // Resize mask to original image size (use NEAREST for masks)
    cv::Mat resized_mask;
    cv::resize(mask, resized_mask, in_image_ptr->image.size(), 0, 0, cv::INTER_NEAREST);
    
    // Publish binary mask as MONO8
    sensor_msgs::msg::Image::SharedPtr out_msg = 
      cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::MONO8, resized_mask).toImageMsg();
    pub_.publish(out_msg);
  }
}

}  // namespace autoware_pov::vision

RCLCPP_COMPONENTS_REGISTER_NODE(autoware_pov::vision::RunModelNode)