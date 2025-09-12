#ifndef ONNX_RUNTIME_BACKEND_HPP_
#define ONNX_RUNTIME_BACKEND_HPP_

#include "inference_backend_base.hpp"
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#define LOG_TYPE_NONE  0
#define LOG_TYPE_ROS   1

#ifndef LOG_TYPE
  #define LOG_TYPE LOG_TYPE_ROS
#endif

#if LOG_TYPE == LOG_TYPE_ROS
  #include "rclcpp/rclcpp.hpp"
  #define LOG_INFO(...) \
    RCLCPP_INFO(rclcpp::get_logger("vision_model_runner"), __VA_ARGS__)  
  #define LOG_WARN(...) \
    RCLCPP_WARN(rclcpp::get_logger("vision_model_runner"), __VA_ARGS__)  
  #define LOG_ERROR(...) \
    RCLCPP_ERROR(rclcpp::get_logger("vision_model_runner"), __VA_ARGS__)
#else
  #define LOG_INFO(...) printf(__VA_ARGS__)
  #define LOG_WARN(...) printf(__VA_ARGS__)
  #define LOG_ERROR(...) printf(__VA_ARGS__)
#endif

namespace autoware_pov::vision
{

class OnnxRuntimeBackend : public InferenceBackend
{
public:
  OnnxRuntimeBackend(const std::string & model_path, const std::string & precision, int gpu_id);
  ~OnnxRuntimeBackend() = default;

  bool doInference(const cv::Mat & input_image) override;
  
  // Only tensor access
  const float* getRawTensorData() const override;
  std::vector<int64_t> getTensorShape() const override;
  
  int getModelInputHeight() const override { return model_input_height_; }
  int getModelInputWidth() const override { return model_input_width_; }

private:
  void initializeOrtSession(const std::string & model_path, const std::string & precision, int gpu_id);
  void preprocess(const cv::Mat & input_image, std::vector<float> & output_tensor, std::vector<int64_t> & input_dims);
  
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> ort_session_;
  Ort::MemoryInfo memory_info_;
  
  std::vector<char*> input_names_;
  std::vector<char*> output_names_;
  std::vector<Ort::Value> last_output_tensors_;
  
  int model_input_height_;
  int model_input_width_;
};

}  // namespace autoware_pov::vision

#endif  // ONNX_RUNTIME_BACKEND_HPP_