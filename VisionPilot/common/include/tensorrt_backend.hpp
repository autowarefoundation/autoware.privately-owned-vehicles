#ifndef TENSORRT_BACKEND_HPP_
#define TENSORRT_BACKEND_HPP_

#include "inference_backend_base.hpp"
#include <NvInfer.h>
#include <memory>
#include <vector>

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
  void getRawOutput(cv::Mat & output, const cv::Size & output_size, const std::string & model_type = "segmentation") const override;

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