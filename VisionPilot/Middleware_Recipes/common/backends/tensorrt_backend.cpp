#include "../include/tensorrt_backend.hpp"
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <numeric>
#include <stdexcept>

#define CUDA_CHECK(status)                         \
  do {                                             \
    auto ret = (status);                           \
    if (ret != 0) {                                \
      LOG_ERROR(                                   \
        "[tensorrt_backend] Cuda failure: %s",     \
        cudaGetErrorString(ret));                  \
      throw std::runtime_error("Cuda failure");    \
    }                                              \
  } while (0)

namespace autoware_pov::vision
{

void Logger::log(Severity severity, const char * msg) noexcept
{
  if (severity <= Severity::kWARNING) {
    if (severity == Severity::kERROR) {
      LOG_ERROR("[tensorrt_backend] %s", msg);
    } else if (severity == Severity::kWARNING) {
      LOG_WARN("[tensorrt_backend] %s", msg);
    } else {
      LOG_INFO("[tensorrt_backend] %s", msg);
    }
  }
}

TensorRTBackend::TensorRTBackend(
  const std::string & model_path, const std::string & precision, int gpu_id)
{
  CUDA_CHECK(cudaSetDevice(gpu_id));

  std::string engine_path = model_path + "." + precision + ".engine";
  std::ifstream engine_file(engine_path, std::ios::binary);

  if (engine_file) {
    LOG_INFO("[tensorrt_backend] Found pre-built %s engine at %s", precision.c_str(), engine_path.c_str());
    loadEngine(engine_path);
  } else {
    LOG_INFO("[tensorrt_backend] No pre-built %s engine found. Building from ONNX model: %s", precision.c_str(), model_path.c_str());
    buildEngineFromOnnx(model_path, precision);
    
    LOG_INFO("[tensorrt_backend] Saving %s engine to %s", precision.c_str(), engine_path.c_str());
    std::unique_ptr<nvinfer1::IHostMemory> model_stream{engine_->serialize()};
    std::ofstream out_file(engine_path, std::ios::binary);
    out_file.write(reinterpret_cast<const char *>(model_stream->data()), model_stream->size());
  }
  
  context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
  if (!context_) {
    throw std::runtime_error("Failed to create TensorRT execution context");
  }

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  stream_ = stream;
  
  const char* input_name = engine_->getIOTensorName(0);
  const char* output_name = engine_->getIOTensorName(1);

  auto input_dims = engine_->getTensorShape(input_name);
  auto output_dims = engine_->getTensorShape(output_name);

  model_input_height_ = input_dims.d[2];
  model_input_width_ = input_dims.d[3];
  
  model_output_classes_ = output_dims.d[1];
  model_output_height_ = output_dims.d[2];
  model_output_width_ = output_dims.d[3];
  
  auto input_vol = std::accumulate(input_dims.d, input_dims.d + input_dims.nbDims, 1LL, std::multiplies<int64_t>());
  auto output_vol = std::accumulate(output_dims.d, output_dims.d + output_dims.nbDims, 1LL, std::multiplies<int64_t>());
  model_output_elem_count_ = output_vol;

  CUDA_CHECK(cudaMalloc(&input_buffer_gpu_, input_vol * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&output_buffer_gpu_, output_vol * sizeof(float)));

  context_->setTensorAddress(input_name, input_buffer_gpu_);
  context_->setTensorAddress(output_name, output_buffer_gpu_);

  output_buffer_host_.resize(model_output_elem_count_);
}

TensorRTBackend::~TensorRTBackend()
{
  cudaFree(input_buffer_gpu_);
  cudaFree(output_buffer_gpu_);
  if (stream_) {
    cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
  }
}

void TensorRTBackend::buildEngineFromOnnx(
  const std::string & onnx_path, const std::string & precision)
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
  
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 30);

  // Set optimization profile (min, opt, max) for dynamic shapes
  nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
  profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 3, 320, 640));
  profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 3, 320, 640));
  profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, 3, 320, 640));
  config->addOptimizationProfile(profile);
  
  if (precision == "fp16" && builder->platformHasFastFp16()) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    LOG_INFO("[tensorrt_backend] Building TensorRT engine with FP16 precision");
  } else {
    LOG_INFO("[tensorrt_backend] Building TensorRT engine with FP32 precision");
  }

  std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
  engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(plan->data(), plan->size()));

  if (!engine_) {
    throw std::runtime_error("Failed to build TensorRT engine.");
  }
}

void TensorRTBackend::loadEngine(const std::string & engine_path)
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

void TensorRTBackend::preprocess(const cv::Mat & input_image, float * buffer)
{
  cv::Mat resized_image, float_image;
  cv::resize(input_image, resized_image, cv::Size(model_input_width_, model_input_height_));
  resized_image.convertTo(float_image, CV_32FC3, 1.0 / 255.0);

  // Use original ImageNet normalization (BGR order from original AUTOSEG)
  cv::subtract(float_image, cv::Scalar(0.406, 0.456, 0.485), float_image);
  cv::divide(float_image, cv::Scalar(0.225, 0.224, 0.229), float_image);
  
  std::vector<cv::Mat> channels(3);
  cv::split(float_image, channels);
  
  // HWC to CHW - exactly like original
  memcpy(buffer, channels[0].data, model_input_height_ * model_input_width_ * sizeof(float));
  memcpy(buffer + model_input_height_ * model_input_width_, channels[1].data, model_input_height_ * model_input_width_ * sizeof(float));
  memcpy(buffer + 2 * model_input_height_ * model_input_width_, channels[2].data, model_input_height_ * model_input_width_ * sizeof(float));
}

bool TensorRTBackend::doInference(const cv::Mat & input_image)
{
  std::vector<float> preprocessed_data(model_input_width_ * model_input_height_ * 3);
  preprocess(input_image, preprocessed_data.data());

  CUDA_CHECK(cudaMemcpyAsync(
    input_buffer_gpu_, preprocessed_data.data(), preprocessed_data.size() * sizeof(float),
    cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream_)));

  bool status = context_->enqueueV3(static_cast<cudaStream_t>(stream_));

  if (!status) {
    LOG_ERROR("[tensorrt_backend] TensorRT inference failed");
    return false;
  }

  CUDA_CHECK(cudaMemcpyAsync(
    output_buffer_host_.data(), output_buffer_gpu_,
    output_buffer_host_.size() * sizeof(float), cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(stream_)));

  CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(stream_)));
  
  return true;
}



const float* TensorRTBackend::getRawTensorData() const
{
    if (output_buffer_host_.empty()) {
        throw std::runtime_error("Inference has not been run yet. Call doInference() first.");
    }
    return output_buffer_host_.data();
}

std::vector<int64_t> TensorRTBackend::getTensorShape() const
{
    return {1, static_cast<int64_t>(model_output_classes_), static_cast<int64_t>(model_output_height_), static_cast<int64_t>(model_output_width_)};
}

}  // namespace autoware_pov::vision