#include "inference/autosteer_engine.hpp"
#include "inference/onnxruntime_session.hpp"
#include <stdexcept>
#include <cstdio>
#include <algorithm>
#include <numeric>

// Simple logging macros (standalone version)
#define LOG_INFO(...) printf("[INFO] "); printf(__VA_ARGS__); printf("\n")
#define LOG_ERROR(...) printf("[ERROR] "); printf(__VA_ARGS__); printf("\n")

namespace autoware_pov::vision::egolanes
{

AutoSteerOnnxEngine::AutoSteerOnnxEngine(
  const std::string& model_path,
  const std::string& provider,
  const std::string& precision,
  int device_id,
  const std::string& cache_dir)
{
  // Create session using factory (reuses same logic as EgoLanes)
  // Use 1GB workspace for AutoSteer with separate cache prefix
  session_ = OnnxRuntimeSessionFactory::createSession(
    model_path, provider, precision, device_id, cache_dir, 1.0, "autosteer_"
  );
  
  // Create memory info for CPU tensors
  memory_info_ = std::make_unique<Ort::MemoryInfo>(
    Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)
  );
  
  // Get input/output names and store them permanently
  Ort::AllocatorWithDefaultOptions allocator;
  
  // Input (typically "input")
  auto input_name_allocated = session_->GetInputNameAllocated(0, allocator);
  input_name_storage_ = std::string(input_name_allocated.get());
  input_names_.push_back(input_name_storage_.c_str());
  
  // Output (typically "output")
  auto output_name_allocated = session_->GetOutputNameAllocated(0, allocator);
  output_name_storage_ = std::string(output_name_allocated.get());
  output_names_.push_back(output_name_storage_.c_str());
  
  // Get input shape: [1, 6, 80, 160]
  auto input_shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  model_input_channels_ = static_cast<int>(input_shape[1]);  // 6
  model_input_height_ = static_cast<int>(input_shape[2]);    // 80
  model_input_width_ = static_cast<int>(input_shape[3]);     // 160
  
  // Get output shape: [1, 61] (steering classes)
  auto output_shape = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  model_output_classes_ = static_cast<int>(output_shape[1]);  // 61
  
  LOG_INFO("[autosteer_engine] AutoSteer engine initialized successfully");
  LOG_INFO("[autosteer_engine] - Input: [1, %d, %d, %d]", 
           model_input_channels_, model_input_height_, model_input_width_);
  LOG_INFO("[autosteer_engine] - Output: [1, %d] steering classes", model_output_classes_);
}

AutoSteerOnnxEngine::~AutoSteerOnnxEngine()
{
  // Smart pointers handle cleanup automatically
}

bool AutoSteerOnnxEngine::doInference(const std::vector<float>& input_buffer)
{
  // Validate input size
  size_t expected_size = model_input_channels_ * model_input_height_ * model_input_width_;
  if (input_buffer.size() != expected_size) {
    LOG_ERROR("[autosteer_engine] Invalid input size: %zu (expected %zu)", 
              input_buffer.size(), expected_size);
    return false;
  }
  
  // Create input tensor from pre-concatenated buffer
  std::vector<int64_t> input_shape = {1, model_input_channels_, model_input_height_, model_input_width_};
  auto input_tensor = Ort::Value::CreateTensor<float>(
    *memory_info_,
    const_cast<float*>(input_buffer.data()),  // ONNX Runtime requires non-const
    input_buffer.size(),
    input_shape.data(),
    input_shape.size()
  );
  
  // Run inference (ONNX Runtime allocates output automatically)
  try {
    output_tensors_ = session_->Run(
      Ort::RunOptions{nullptr},
      input_names_.data(),
      &input_tensor,
      1,
      output_names_.data(),
      1
    );
  } catch (const Ort::Exception& e) {
    LOG_ERROR("[autosteer_engine] Inference failed: %s", e.what());
    return false;
  }
  
  return true;
}

float AutoSteerOnnxEngine::postProcess()
{
  if (output_tensors_.empty()) {
    LOG_ERROR("[autosteer_engine] No output tensors available");
    return 0.0f;
  }
  
  const float* raw_output = output_tensors_[0].GetTensorData<float>();
  
  // Find argmax (class with highest probability)
  // Output format: [1, 61] logits
  int max_class = 0;
  float max_value = raw_output[0];
  
  for (int i = 1; i < model_output_classes_; ++i) {
    if (raw_output[i] > max_value) {
      max_value = raw_output[i];
      max_class = i;
    }
  }
  
  // Convert class to steering angle: argmax - 30
  // Classes: 0-60 → Angles: -30 to +30 degrees
  float steering_angle = static_cast<float>(max_class - 30);
  
  return steering_angle;
}

float AutoSteerOnnxEngine::inference(const std::vector<float>& concat_input)
{
  // Run inference
  if (!doInference(concat_input)) {
    LOG_ERROR("[autosteer_engine] Inference failed");
    return 0.0f;
  }
  
  // Post-process and return steering angle
  return postProcess();
}

}  // namespace autoware_pov::vision::egolanes

