#ifndef AUTOWARE_POV__AUTOSEG__SCENE_SEG__TENSORRT_BACKEND_HPP_
#define AUTOWARE_POV__AUTOSEG__SCENE_SEG__TENSORRT_BACKEND_HPP_

#include "inference_backend.hpp"

#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <opencv2/opencv.hpp>

namespace autoware_pov::AutoSeg::SceneSeg
{

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger
{
  void log(Severity severity, const char * msg) noexcept override;
};

class TensorRTBackend : public InferenceBackend
{
public:
  explicit TensorRTBackend(const std::string & model_path, const std::string & precision, int gpu_id);
  ~TensorRTBackend() override;

  bool doInference(const cv::Mat & input_image) override;
  void getRawMask(cv::Mat & raw_mask, const cv::Size & output_size) const override;

private:
  void buildEngineFromOnnx(const std::string & onnx_path, const std::string & precision);
  void loadEngine(const std::string & engine_path);
  void preprocess(const cv::Mat & input_image, float * buffer);

  Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_{nullptr};
  std::unique_ptr<nvinfer1::ICudaEngine> engine_{nullptr};
  std::unique_ptr<nvinfer1::IExecutionContext> context_{nullptr};

  // GPU memory
  void * input_buffer_gpu_{nullptr};
  void * output_buffer_gpu_{nullptr};
  
  // Host memory for output
  mutable std::vector<float> output_buffer_host_;

  // CUDA stream
  void * stream_{nullptr}; // Using void* to avoid including cuda_runtime.h in header

  // Model input/output details
  int model_input_width_{0};
  int model_input_height_{0};
  int model_output_elem_count_{0};
  int model_output_classes_{0};
  int model_output_height_{0};
  int model_output_width_{0};

};

}  // namespace autoware_pov::AutoSeg::SceneSeg

#endif  // AUTOWARE_POV__AUTOSEG__SCENE_SEG__TENSORRT_BACKEND_HPP_ 