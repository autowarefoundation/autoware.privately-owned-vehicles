#include "../include/onnx_runtime_backend.hpp"
#include <vector>
#include <stdexcept>
//#include "rclcpp/rclcpp.hpp"
#include "onnxruntime_c_api.h"

namespace autoware_pov::vision
{

OnnxRuntimeBackend::OnnxRuntimeBackend(const std::string & model_path, const std::string & precision, int gpu_id)
: env_(ORT_LOGGING_LEVEL_WARNING, "autoware_pov_vision"), memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
  initializeOrtSession(model_path, precision, gpu_id);
}

void OnnxRuntimeBackend::initializeOrtSession(const std::string& model_path, const std::string& precision, int gpu_id)
{
  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  if (precision == "cuda") {
     try {
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options_, gpu_id));
        //RCLCPP_INFO(rclcpp::get_logger("vision_model_runner"), "Using CUDA Execution Provider on GPU %d.", gpu_id);
     } catch (const Ort::Exception& e) {
        //RCLCPP_WARN(rclcpp::get_logger("vision_model_runner"), "CUDA Execution Provider is not available. Falling back to CPU. Error: %s", e.what());
     }
  } else {
    //RCLCPP_INFO(rclcpp::get_logger("vision_model_runner"), "Using default CPU Execution Provider.");
  }

  ort_session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);

  Ort::AllocatorWithDefaultOptions allocator;
  input_names_.push_back(strdup(ort_session_->GetInputNameAllocated(0, allocator).get()));
  output_names_.push_back(strdup(ort_session_->GetOutputNameAllocated(0, allocator).get()));
  
  auto input_shape = ort_session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  model_input_height_ = static_cast<int>(input_shape[2]);
  model_input_width_ = static_cast<int>(input_shape[3]);
}

void OnnxRuntimeBackend::preprocess(const cv::Mat & input_image, std::vector<float> & output_tensor, std::vector<int64_t>& input_dims)
{
    cv::Mat resized_image, float_image;
    cv::resize(input_image, resized_image, cv::Size(model_input_width_, model_input_height_));
    resized_image.convertTo(float_image, CV_32FC3, 1.0 / 255.0);

    // Use original ImageNet normalization (BGR order from original AUTOSEG)
    cv::subtract(float_image, cv::Scalar(0.406, 0.456, 0.485), float_image);
    cv::divide(float_image, cv::Scalar(0.225, 0.224, 0.229), float_image);

    output_tensor.resize(model_input_height_ * model_input_width_ * 3);
    std::vector<cv::Mat> channels(3);
    cv::split(float_image, channels);
    // HWC to CHW - exactly like original
    memcpy(output_tensor.data(), channels[0].data, model_input_height_ * model_input_width_ * sizeof(float));
    memcpy(output_tensor.data() + model_input_height_ * model_input_width_, channels[1].data, model_input_height_ * model_input_width_ * sizeof(float));
    memcpy(output_tensor.data() + 2 * model_input_height_ * model_input_width_, channels[2].data, model_input_height_ * model_input_width_ * sizeof(float));
    
    input_dims = {1, 3, model_input_height_, model_input_width_};
}

bool OnnxRuntimeBackend::doInference(const cv::Mat & input_image)
{
    std::vector<float> preprocessed_data;
    std::vector<int64_t> input_dims;
    preprocess(input_image, preprocessed_data, input_dims);

    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, preprocessed_data.data(), preprocessed_data.size(),
        input_dims.data(), input_dims.size());

    try {
        last_output_tensors_ = ort_session_->Run(
            Ort::RunOptions{nullptr}, const_cast<const char* const*>(input_names_.data()), &input_tensor, 1,
            const_cast<const char* const*>(output_names_.data()), output_names_.size());
    } catch (const Ort::Exception& e) {
        //RCLCPP_ERROR(rclcpp::get_logger("vision_model_runner"), "ONNX Runtime inference failed: %s", e.what());
        return false;
    }
    
    return true;
}

void OnnxRuntimeBackend::getRawOutput(cv::Mat & output, const cv::Size & output_size, const std::string & model_type) const
{
    if (last_output_tensors_.empty()) {
        throw std::runtime_error("Inference has not been run yet. Call doInference() first.");
    }
    
    const float* output_data = last_output_tensors_.front().GetTensorData<float>();
    auto output_dims = last_output_tensors_.front().GetTensorTypeAndShapeInfo().GetShape();
    
    const int height = static_cast<int>(output_dims[2]);
    const int width = static_cast<int>(output_dims[3]);
    const int channels = static_cast<int>(output_dims[1]);

    if (model_type == "depth") {
        // Depth estimation: return raw float values without processing
        cv::Mat raw_output(height, width, CV_32FC1, const_cast<float*>(output_data));
        cv::resize(raw_output, output, output_size, 0, 0, cv::INTER_LINEAR);
        
    } else if (channels > 1) {
        // Multi-class segmentation: do argmax like original scene seg
        cv::Mat argmax_mask(height, width, CV_8UC1);
        
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                float max_score = -std::numeric_limits<float>::infinity();
                uint8_t best_class = 0;
                
                for (int c = 0; c < channels; ++c) {
                    float score = output_data[c * height * width + h * width + w];
                    if (score > max_score) {
                        max_score = score;
                        best_class = static_cast<uint8_t>(c);
                    }
                }
                
                argmax_mask.at<uint8_t>(h, w) = best_class;
            }
        }
        
        // Resize using nearest neighbor to preserve class boundaries
        cv::resize(argmax_mask, output, output_size, 0, 0, cv::INTER_NEAREST);
        
    } else {
        // Binary segmentation: threshold like original domain seg
        cv::Mat raw_output(height, width, CV_32FC1, const_cast<float*>(output_data));
        
        cv::Mat thresholded;
        cv::threshold(raw_output, thresholded, 0.0, 1.0, cv::THRESH_BINARY);
        
        cv::Mat binary_mask;
        thresholded.convertTo(binary_mask, CV_8UC1, 255.0);
        
        // Resize using nearest neighbor
        cv::resize(binary_mask, output, output_size, 0, 0, cv::INTER_NEAREST);
    }
}

}  // namespace autoware_pov::vision