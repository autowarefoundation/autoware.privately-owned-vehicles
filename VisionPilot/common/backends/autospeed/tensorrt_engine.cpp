#include "tensorrt_engine.hpp"
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include <cstdlib>

#define CUDA_CHECK(status)                         \
  do {                                             \
    auto ret = (status);                           \
    if (ret != 0) {                                \
      LOG_ERROR(                                   \
        "[autospeed_trt] CUDA failure: %s",       \
        cudaGetErrorString(ret));                  \
      throw std::runtime_error("CUDA failure");    \
    }                                              \
  } while (0)

namespace autoware_pov::vision::autospeed
{

void Logger::log(Severity severity, const char * msg) noexcept
{
  if (severity <= Severity::kWARNING) {
    if (severity == Severity::kERROR) {
      LOG_ERROR("[autospeed_trt] %s", msg);
    } else if (severity == Severity::kWARNING) {
      LOG_WARN("[autospeed_trt] %s", msg);
    } else {
      LOG_INFO("[autospeed_trt] %s", msg);
    }
  }
}

AutoSpeedTensorRTEngine::AutoSpeedTensorRTEngine(
  const std::string & model_path, 
  const std::string & precision, 
  int gpu_id)
{
  CUDA_CHECK(cudaSetDevice(gpu_id));

  std::string onnx_path = model_path;
  
  // Check if input is PyTorch checkpoint
  if (model_path.substr(model_path.find_last_of(".") + 1) == "pt") {
    LOG_INFO("[autospeed_trt] Detected PyTorch checkpoint: %s", model_path.c_str());
    onnx_path = model_path.substr(0, model_path.find_last_of(".")) + ".onnx";
    
    // Check if ONNX already exists
    std::ifstream onnx_check(onnx_path);
    if (!onnx_check) {
      LOG_INFO("[autospeed_trt] Converting PyTorch to ONNX...");
      convertPyTorchToOnnx(model_path, onnx_path);
    } else {
      LOG_INFO("[autospeed_trt] Found existing ONNX model: %s", onnx_path.c_str());
    }
  }

  // Now handle ONNX → TensorRT engine
  std::string engine_path = onnx_path + "." + precision + ".engine";
  std::ifstream engine_file(engine_path, std::ios::binary);

  if (engine_file) {
    LOG_INFO("[autospeed_trt] Found pre-built %s engine at %s", precision.c_str(), engine_path.c_str());
    loadEngine(engine_path);
  } else {
    LOG_INFO("[autospeed_trt] No pre-built %s engine found. Building from ONNX: %s", 
             precision.c_str(), onnx_path.c_str());
    buildEngineFromOnnx(onnx_path, precision);
    
    LOG_INFO("[autospeed_trt] Saving %s engine to %s", precision.c_str(), engine_path.c_str());
    std::unique_ptr<nvinfer1::IHostMemory> model_stream{engine_->serialize()};
    std::ofstream out_file(engine_path, std::ios::binary);
    out_file.write(reinterpret_cast<const char *>(model_stream->data()), model_stream->size());
  }
  
  // Create execution context
  context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
  if (!context_) {
    throw std::runtime_error("Failed to create TensorRT execution context");
  }

  // Create CUDA stream
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  stream_ = stream;
  
  // Get input/output tensor information
  const char* input_name = engine_->getIOTensorName(0);
  const char* output_name = engine_->getIOTensorName(1);

  auto input_dims = engine_->getTensorShape(input_name);
  auto output_dims = engine_->getTensorShape(output_name);

  model_input_height_ = input_dims.d[2];
  model_input_width_ = input_dims.d[3];
  
  // AutoSpeed output format: [1, num_predictions, num_attributes]
  // e.g., [1, 8400, 85] for YOLO-like models (4 bbox + 1 obj + 80 classes)
  model_output_predictions_ = output_dims.d[1];
  model_output_channels_ = output_dims.d[2];
  
  auto input_vol = std::accumulate(input_dims.d, input_dims.d + input_dims.nbDims, 1LL, std::multiplies<int64_t>());
  auto output_vol = std::accumulate(output_dims.d, output_dims.d + output_dims.nbDims, 1LL, std::multiplies<int64_t>());
  model_output_elem_count_ = output_vol;

  // Allocate GPU buffers
  CUDA_CHECK(cudaMalloc(&input_buffer_gpu_, input_vol * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&output_buffer_gpu_, output_vol * sizeof(float)));

  // Set tensor addresses
  context_->setTensorAddress(input_name, input_buffer_gpu_);
  context_->setTensorAddress(output_name, output_buffer_gpu_);

  // Allocate host output buffer
  output_buffer_host_.resize(model_output_elem_count_);
  
  LOG_INFO("[autospeed_trt] Engine initialized successfully");
  LOG_INFO("[autospeed_trt] Input: %dx%d, Output: %d predictions x %d attributes", 
           model_input_width_, model_input_height_, 
           model_output_predictions_, model_output_channels_);
}

AutoSpeedTensorRTEngine::~AutoSpeedTensorRTEngine()
{
  cudaFree(input_buffer_gpu_);
  cudaFree(output_buffer_gpu_);
  if (stream_) {
    cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
  }
}

void AutoSpeedTensorRTEngine::convertPyTorchToOnnx(
  const std::string & pytorch_path, 
  const std::string & onnx_path)
{
  // Call Python script to convert PyTorch → ONNX
  // This assumes you have a conversion script in your project
  std::string convert_script = "python3 -c \"import torch; "
    "model = torch.load('" + pytorch_path + "', map_location='cpu', weights_only=False)['model']; "
    "dummy_input = torch.randn(1, 3, 640, 640); "
    "torch.onnx.export(model, dummy_input, '" + onnx_path + "', "
    "opset_version=11, input_names=['input'], output_names=['output'])\"";
  
  int result = std::system(convert_script.c_str());
  if (result != 0) {
    throw std::runtime_error("Failed to convert PyTorch to ONNX. Ensure PyTorch is installed.");
  }
  
  LOG_INFO("[autospeed_trt] Successfully converted PyTorch to ONNX: %s", onnx_path.c_str());
}

void AutoSpeedTensorRTEngine::buildEngineFromOnnx(
  const std::string & onnx_path, 
  const std::string & precision)
{
  runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger_));
  
  auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger_));
  auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  
  const auto explicitBatch =
    1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network =
    std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
  
  auto parser =
    std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger_));
  
  if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    throw std::runtime_error("Failed to parse ONNX file.");
  }
  
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30); // 1GB

  // Set optimization profile for AutoSpeed (640x640 input)
  nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
  profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 3, 640, 640));
  profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 3, 640, 640));
  profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, 3, 640, 640));
  config->addOptimizationProfile(profile);
  
  if (precision == "fp16" && builder->platformHasFastFp16()) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    LOG_INFO("[autospeed_trt] Building TensorRT engine with FP16 precision");
  } else {
    LOG_INFO("[autospeed_trt] Building TensorRT engine with FP32 precision");
  }

  std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
  engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(plan->data(), plan->size()));

  if (!engine_) {
    throw std::runtime_error("Failed to build TensorRT engine.");
  }
}

void AutoSpeedTensorRTEngine::loadEngine(const std::string & engine_path)
{
  std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(size);
  file.read(buffer.data(), size);
  
  runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger_));
  engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
    runtime_->deserializeCudaEngine(buffer.data(), buffer.size()));
  if (!engine_) {
    throw std::runtime_error("Failed to load TensorRT engine.");
  }
}

void AutoSpeedTensorRTEngine::preprocessAutoSpeed(const cv::Mat & input_image, float * buffer)
{
  // Store original dimensions for later coordinate transformation
  orig_width_ = input_image.cols;
  orig_height_ = input_image.rows;

  // Step 1: Letterbox resize to 640x640 (maintain aspect ratio with padding)
  int target_w = model_input_width_;
  int target_h = model_input_height_;
  
  scale_ = std::min(
    static_cast<float>(target_w) / orig_width_,
    static_cast<float>(target_h) / orig_height_
  );
  
  int new_w = static_cast<int>(orig_width_ * scale_);
  int new_h = static_cast<int>(orig_height_ * scale_);
  
  cv::Mat resized;
  cv::resize(input_image, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
  
  // Step 2: Create padded image with gray color (114, 114, 114)
  cv::Mat padded(target_h, target_w, CV_8UC3, cv::Scalar(114, 114, 114));
  
  pad_x_ = (target_w - new_w) / 2;
  pad_y_ = (target_h - new_h) / 2;
  
  resized.copyTo(padded(cv::Rect(pad_x_, pad_y_, new_w, new_h)));
  
  // Step 3: Convert to float and normalize to [0, 1] (NOT ImageNet normalization!)
  cv::Mat float_image;
  padded.convertTo(float_image, CV_32FC3, 1.0 / 255.0);
  
  // Step 4: Convert BGR to RGB and HWC to CHW format
  std::vector<cv::Mat> channels(3);
  cv::split(float_image, channels);
  
  // BGR → RGB: Reverse channel order
  // AutoSpeed expects RGB, OpenCV loads as BGR
  int channel_size = target_h * target_w;
  memcpy(buffer, channels[2].data, channel_size * sizeof(float));  // R
  memcpy(buffer + channel_size, channels[1].data, channel_size * sizeof(float));  // G
  memcpy(buffer + 2 * channel_size, channels[0].data, channel_size * sizeof(float));  // B
}

bool AutoSpeedTensorRTEngine::doInference(const cv::Mat & input_image)
{
  // Allocate preprocessed data buffer
  std::vector<float> preprocessed_data(model_input_width_ * model_input_height_ * 3);
  
  // Preprocess with letterbox
  preprocessAutoSpeed(input_image, preprocessed_data.data());

  // Copy to GPU
  CUDA_CHECK(cudaMemcpyAsync(
    input_buffer_gpu_, preprocessed_data.data(), preprocessed_data.size() * sizeof(float),
    cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream_)));

  // Run inference
  bool status = context_->enqueueV3(static_cast<cudaStream_t>(stream_));

  if (!status) {
    LOG_ERROR("[autospeed_trt] TensorRT inference failed");
    return false;
  }

  // Copy output back to host
  CUDA_CHECK(cudaMemcpyAsync(
    output_buffer_host_.data(), output_buffer_gpu_,
    output_buffer_host_.size() * sizeof(float), cudaMemcpyDeviceToHost, 
    static_cast<cudaStream_t>(stream_)));

  CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(stream_)));
  
  return true;
}

const float* AutoSpeedTensorRTEngine::getRawTensorData() const
{
  if (output_buffer_host_.empty()) {
    throw std::runtime_error("Inference has not been run yet. Call doInference() first.");
  }
  return output_buffer_host_.data();
}

std::vector<int64_t> AutoSpeedTensorRTEngine::getTensorShape() const
{
  // Return shape: [batch=1, num_predictions, num_attributes]
  return {1, static_cast<int64_t>(model_output_predictions_), static_cast<int64_t>(model_output_channels_)};
}

}  // namespace autoware_pov::vision::autospeed

