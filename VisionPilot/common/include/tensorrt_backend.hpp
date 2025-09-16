#ifndef TENSORRT_BACKEND_HPP_
#define TENSORRT_BACKEND_HPP_

#include "inference_backend_base.hpp"
#include <NvInfer.h>
#include <memory>
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

class Logger : public nvinfer1::ILogger
{
  void log(Severity severity, const char * msg) noexcept override;
};

class TensorRTBackend : public InferenceBackend
{
public:
  TensorRTBackend(const std::string & model_path, const std::string & precision, int gpu_id);
  ~TensorRTBackend();

  bool doInference(const cv::Mat & input_image) override;

  // Only tensor access
  const float* getRawTensorData() const override;
  std::vector<int64_t> getTensorShape() const override;

  int getModelInputHeight() const override { return model_input_height_; }
  int getModelInputWidth() const override { return model_input_width_; }

private:
  void buildEngineFromOnnx(const std::string & onnx_path, const std::string & precision);
  void loadEngine(const std::string & engine_path);
  void preprocess(const cv::Mat & input_image, float * buffer);

  Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_{nullptr};
  std::unique_ptr<nvinfer1::ICudaEngine> engine_{nullptr};
  std::unique_ptr<nvinfer1::IExecutionContext> context_{nullptr};

  void* stream_{nullptr};
  void* input_buffer_gpu_{nullptr};
  void* output_buffer_gpu_{nullptr};
  std::vector<float> output_buffer_host_;

  int model_input_height_;
  int model_input_width_;
  int model_output_height_;
  int model_output_width_;
  int model_output_classes_;
  int64_t model_output_elem_count_;
};

}  // namespace autoware_pov::vision

#endif  // TENSORRT_BACKEND_HPP_