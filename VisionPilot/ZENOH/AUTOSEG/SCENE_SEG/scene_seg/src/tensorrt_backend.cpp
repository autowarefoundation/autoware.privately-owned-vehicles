#include "tensorrt_backend.hpp"
//#include "rclcpp/rclcpp.hpp"

#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <numeric>
#include <stdexcept>

// Helper to check for CUDA errors
#define CUDA_CHECK(status)                                           \
  do {                                                               \
    auto ret = (status);                                             \
    if (ret != 0) {                                                  \
      throw std::runtime_error("Cuda failure");                      \
    }                                                                \
  } while (0)
//#define CUDA_CHECK(status)                                           \
//  do {                                                               \
//    auto ret = (status);                                             \
//    if (ret != 0) {                                                  \
//      RCLCPP_ERROR(                                                  \
//        rclcpp::get_logger("tensorrt_backend"), "Cuda failure: %s",   \
//        cudaGetErrorString(ret));                                    \
//      throw std::runtime_error("Cuda failure");                      \
//    }                                                                \
//  } while (0)

namespace autoware_pov::AutoSeg::SceneSeg
{

void Logger::log(Severity severity, const char * msg) noexcept
{
  if (severity <= Severity::kWARNING) {
    if (severity == Severity::kERROR) {
      //RCLCPP_ERROR(rclcpp::get_logger("tensorrt_backend"), "%s", msg);
    } else if (severity == Severity::kWARNING) {
      //RCLCPP_WARN(rclcpp::get_logger("tensorrt_backend"), "%s", msg);
    } else {
      //RCLCPP_INFO(rclcpp::get_logger("tensorrt_backend"), "%s", msg);
    }
  }
}

TensorRTBackend::TensorRTBackend(
  const std::string & model_path, const std::string & precision, int gpu_id)
{
  CUDA_CHECK(cudaSetDevice(gpu_id));

  // Check if a pre-built engine file exists (with precision-specific naming)
  std::string engine_path = model_path + "." + precision + ".engine";
  std::ifstream engine_file(engine_path, std::ios::binary);

  if (engine_file) {
    //RCLCPP_INFO(
    //  rclcpp::get_logger("tensorrt_backend"), "Found pre-built %s engine at %s", 
    //  precision.c_str(), engine_path.c_str());
    loadEngine(engine_path);
  } else {
    //RCLCPP_INFO(
    //  rclcpp::get_logger("tensorrt_backend"),
    //  "No pre-built %s engine found. Building from ONNX model: %s", 
    //  precision.c_str(), model_path.c_str());
    buildEngineFromOnnx(model_path, precision);
    
    // Save the engine for future runs with precision-specific naming
    //RCLCPP_INFO(rclcpp::get_logger("tensorrt_backend"), "Saving %s engine to %s", 
    //            precision.c_str(), engine_path.c_str());
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
  
  // With explicit batch, we use I/O Tensors. This logic assumes 1 input and 1 output.
  if (engine_->getNbIOTensors() != 2) {
      throw std::runtime_error("This backend expects the model to have 1 input and 1 output tensor.");
  }
  const char* tensor_name_1 = engine_->getIOTensorName(0);
  const char* tensor_name_2 = engine_->getIOTensorName(1);
  const char* input_name = (engine_->getTensorIOMode(tensor_name_1) == nvinfer1::TensorIOMode::kINPUT) ? tensor_name_1 : tensor_name_2;
  const char* output_name = (engine_->getTensorIOMode(tensor_name_1) == nvinfer1::TensorIOMode::kOUTPUT) ? tensor_name_1 : tensor_name_2;

  auto input_dims = engine_->getTensorShape(input_name);
  auto output_dims = engine_->getTensorShape(output_name);

  // Store model dimensions
  model_input_height_ = input_dims.d[2];
  model_input_width_ = input_dims.d[3];
  
  model_output_classes_ = output_dims.d[1];
  model_output_height_ = output_dims.d[2];
  model_output_width_ = output_dims.d[3];
  
  // Calculate buffer sizes
  auto input_vol = std::accumulate(input_dims.d, input_dims.d + input_dims.nbDims, 1LL, std::multiplies<int64_t>());
  auto output_vol = std::accumulate(output_dims.d, output_dims.d + output_dims.nbDims, 1LL, std::multiplies<int64_t>());
  model_output_elem_count_ = output_vol;

  // Allocate GPU memory
  CUDA_CHECK(cudaMalloc(&input_buffer_gpu_, input_vol * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&output_buffer_gpu_, output_vol * sizeof(float)));

  // Set tensor addresses for enqueueV3
  context_->setTensorAddress(input_name, input_buffer_gpu_);
  context_->setTensorAddress(output_name, output_buffer_gpu_);

  // Allocate host memory
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
  
  // Set precision and other configurations
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 30);  // 1GB
  
  // Handle dynamic shapes by creating an optimization profile
  auto profile = builder->createOptimizationProfile();
  for (int i = 0; i < network->getNbInputs(); ++i) {
    auto input = network->getInput(i);
    auto inputDims = input->getDimensions();
    auto inputName = input->getName();
    
    // For dynamic dimensions, we need to set min, opt, and max shapes
    // Assuming typical segmentation input: [batch, channels, height, width]
    if (inputDims.d[0] == -1) inputDims.d[0] = 1;  // Fix batch size to 1
    if (inputDims.d[2] == -1) inputDims.d[2] = model_input_height_;  // Fix height
    if (inputDims.d[3] == -1) inputDims.d[3] = model_input_width_;   // Fix width
    
    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, inputDims);
    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, inputDims);
    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, inputDims);
    
    //RCLCPP_INFO(rclcpp::get_logger("tensorrt_backend"), 
    //            "Set optimization profile for input '%s': [%d, %d, %d, %d]", 
    //            inputName, inputDims.d[0], inputDims.d[1], inputDims.d[2], inputDims.d[3]);
  }
  config->addOptimizationProfile(profile);
  
  if (precision == "fp16" && builder->platformHasFastFp16()) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    //RCLCPP_INFO(rclcpp::get_logger("tensorrt_backend"), "Building TensorRT engine with FP16 precision");
  } else {
    //RCLCPP_INFO(rclcpp::get_logger("tensorrt_backend"), "Building TensorRT engine with FP32 precision");
  }

  engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
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
  // This is a CPU-based preprocess. For max performance, this should be a CUDA kernel.
  cv::Mat resized_image, float_image;
  cv::resize(input_image, resized_image, cv::Size(model_input_width_, model_input_height_));
  resized_image.convertTo(float_image, CV_32FC3, 1.0 / 255.0);

  cv::subtract(float_image, cv::Scalar(0.406, 0.456, 0.485), float_image);
  cv::divide(float_image, cv::Scalar(0.225, 0.224, 0.229), float_image);
  
  std::vector<cv::Mat> channels(3);
  cv::split(float_image, channels);
  
  // HWC to CHW
  memcpy(buffer, channels[0].data, model_input_width_ * model_input_height_ * sizeof(float));
  memcpy(buffer + model_input_width_ * model_input_height_, channels[1].data, model_input_width_ * model_input_height_ * sizeof(float));
  memcpy(buffer + 2 * model_input_width_ * model_input_height_, channels[2].data, model_input_width_ * model_input_height_ * sizeof(float));
}

bool TensorRTBackend::doInference(const cv::Mat & input_image)
{
  std::vector<float> preprocessed_data(model_input_width_ * model_input_height_ * 3);
  preprocess(input_image, preprocessed_data.data());

  // Copy data to GPU
  CUDA_CHECK(cudaMemcpyAsync(
    input_buffer_gpu_, preprocessed_data.data(), preprocessed_data.size() * sizeof(float),
    cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream_)));

  // Run inference with enqueueV3
  bool status = context_->enqueueV3(static_cast<cudaStream_t>(stream_));

  if (!status) {
    RCLCPP_ERROR(rclcpp::get_logger("tensorrt_backend"), "TensorRT inference failed");
    return false;
  }

  // Copy results back to host
  CUDA_CHECK(cudaMemcpyAsync(
    output_buffer_host_.data(), output_buffer_gpu_,
    output_buffer_host_.size() * sizeof(float), cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(stream_)));

  CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(stream_)));
  
  return true;
}

void TensorRTBackend::getRawMask(cv::Mat & raw_mask, const cv::Size & output_size) const
{
  // Argmax on CPU. For max performance, this should be a CUDA kernel.
  cv::Mat argmax_mask(model_output_height_, model_output_width_, CV_8UC1);

  for (int h = 0; h < model_output_height_; ++h) {
    for (int w = 0; w < model_output_width_; ++w) {
      float max_score = -std::numeric_limits<float>::infinity();
      uint8_t best_class = 0;

      for (int c = 0; c < model_output_classes_; ++c) {
        float score = output_buffer_host_[c * model_output_height_ * model_output_width_ + h * model_output_width_ + w];
        if (score > max_score) {
          max_score = score;
          best_class = static_cast<uint8_t>(c);
        }
      }
      argmax_mask.at<uint8_t>(h, w) = best_class;
    }
  }

  cv::resize(argmax_mask, raw_mask, output_size, 0, 0, cv::INTER_NEAREST);
}

}  // namespace autoware_pov::AutoSeg::SceneSeg 