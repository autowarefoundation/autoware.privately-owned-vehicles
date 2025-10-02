#ifndef AUTOSPEED_TENSORRT_ENGINE_HPP_
#define AUTOSPEED_TENSORRT_ENGINE_HPP_

#include "../../include/inference_backend_base.hpp"
#include <NvInfer.h>
#include <memory>
#include <vector>
#include "../../include/logging.hpp"

namespace autoware_pov::vision::autospeed
{

class Logger : public nvinfer1::ILogger
{
  void log(Severity severity, const char * msg) noexcept override;
};

// AutoSpeed-specific detection structure
struct Detection {
  float x1, y1, x2, y2;  // Bounding box corners
  float confidence;       // Detection confidence
  int class_id;          // Object class ID
};

class AutoSpeedTensorRTEngine : public InferenceBackend
{
public:
  /**
   * @brief Construct AutoSpeed TensorRT Engine
   * @param model_path Path to ONNX model or PyTorch checkpoint (.pt)
   * @param precision "fp32" or "fp16"
   * @param gpu_id GPU device ID
   */
  AutoSpeedTensorRTEngine(
    const std::string & model_path, 
    const std::string & precision, 
    int gpu_id
  );
  
  ~AutoSpeedTensorRTEngine();

  // InferenceBackend interface
  bool doInference(const cv::Mat & input_image) override;
  const float* getRawTensorData() const override;
  std::vector<int64_t> getTensorShape() const override;
  int getModelInputHeight() const override { return model_input_height_; }
  int getModelInputWidth() const override { return model_input_width_; }

  // AutoSpeed-specific methods
  /**
   * @brief Get letterbox transformation parameters from last inference
   */
  void getLetterboxParams(float& scale, int& pad_x, int& pad_y) const {
    scale = scale_;
    pad_x = pad_x_;
    pad_y = pad_y_;
  }

  /**
   * @brief Get original image dimensions from last inference
   */
  void getOriginalImageSize(int& width, int& height) const {
    width = orig_width_;
    height = orig_height_;
  }

private:
  // Engine management
  void buildEngineFromOnnx(const std::string & onnx_path, const std::string & precision);
  void loadEngine(const std::string & engine_path);
  void convertPyTorchToOnnx(const std::string & pytorch_path, const std::string & onnx_path);
  
  // AutoSpeed-specific preprocessing (letterbox + normalize to [0,1])
  void preprocessAutoSpeed(const cv::Mat & input_image, float * buffer);

  // TensorRT components
  Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_{nullptr};
  std::unique_ptr<nvinfer1::ICudaEngine> engine_{nullptr};
  std::unique_ptr<nvinfer1::IExecutionContext> context_{nullptr};

  // CUDA resources
  void* stream_{nullptr};
  void* input_buffer_gpu_{nullptr};
  void* output_buffer_gpu_{nullptr};
  std::vector<float> output_buffer_host_;

  // Model dimensions
  int model_input_height_{640};   // AutoSpeed default: 640x640
  int model_input_width_{640};
  int model_output_channels_;     // Number of detection attributes (85 for YOLO-like)
  int model_output_predictions_;  // Number of predictions (e.g., 8400)
  int64_t model_output_elem_count_;

  // Letterbox transformation parameters (saved during preprocessing)
  float scale_{1.0f};
  int pad_x_{0};
  int pad_y_{0};
  int orig_width_{0};
  int orig_height_{0};
};

}  // namespace autoware_pov::vision::autospeed

#endif  // AUTOSPEED_TENSORRT_ENGINE_HPP_

