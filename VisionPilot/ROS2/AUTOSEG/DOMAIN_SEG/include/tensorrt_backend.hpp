#pragma once

#include "inference_backend.hpp"
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>

namespace autoware_pov::AutoSeg::DomainSeg
{

class Logger : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char * msg) noexcept override;
};

class TensorRTBackend : public InferenceBackend
{
public:
  TensorRTBackend(const std::string & model_path, const std::string & precision, int gpu_id);
  ~TensorRTBackend();

  bool doInference(const cv::Mat & input_image) override;
  void getDomainMask(cv::Mat & domain_mask, const cv::Size & output_size) const override;
  
  int getModelInputHeight() const override { return model_input_height_; }
  int getModelInputWidth() const override { return model_input_width_; }

private:
  void buildEngineFromOnnx(const std::string & onnx_path, const std::string & precision);
  void loadEngine(const std::string & engine_path);
  void preprocess(const cv::Mat & input_image, float * buffer);

  Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  
  void * stream_;
  void * input_buffer_gpu_;
  void * output_buffer_gpu_;
  std::vector<float> output_buffer_host_;
  
  int model_input_height_;
  int model_input_width_;
  int model_output_height_;
  int model_output_width_;
  size_t model_output_elem_count_;
};

}  // namespace autoware_pov::AutoSeg::DomainSeg 