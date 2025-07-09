
#ifndef AUTOWARE_POV__AUTOSEG__SCENE_SEG__SCENE_SEG_HPP_
#define AUTOWARE_POV__AUTOSEG__SCENE_SEG__SCENE_SEG_HPP_

#include <string>
#include <vector>
#include <memory>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

namespace autoware_pov::AutoSeg::SceneSeg
{

class SceneSeg
{
public:
  explicit SceneSeg(const std::string & model_path, const std::string & precision, int gpu_id);

  bool doInference(const cv::Mat & input_image);
  void getRawMask(cv::Mat & raw_mask, const cv::Size & output_size) const;
  void colorizeMask(const cv::Mat & raw_mask, const cv::Mat & original_image, cv::Mat & blended_image) const;

private:
  void preprocess(const cv::Mat & input_image, std::vector<float> & output_tensor, std::vector<int64_t>& input_dims);
  void createColorMap();
  void initializeOrtSession(const std::string& model_path, const std::string& precision, int gpu_id);

  // ONNX Runtime session members
  Ort::Env env_;
  std::unique_ptr<Ort::Session> ort_session_{nullptr};
  Ort::MemoryInfo memory_info_{nullptr};
  Ort::SessionOptions session_options_;
  
  // Stored output from the last inference run
  std::vector<Ort::Value> last_output_tensors_;

  // ONNX Model I/O details
  std::vector<char*> input_names_;
  std::vector<char*> output_names_;
  int model_input_width_{0};
  int model_input_height_{0};

  // Segmentation color map for visualization
  std::vector<cv::Vec3b> color_map_;
};

}  // namespace autoware_pov::AutoSeg::SceneSeg

#endif  // AUTOWARE_POV__AUTOSEG__SCENE_SEG__SCENE_SEG_HPP_ 