
#ifndef AUTOWARE_POV__AUTOSEG__SCENE_SEG__SCENE_SEG_HPP_
#define AUTOWARE_POV__AUTOSEG__SCENE_SEG__SCENE_SEG_HPP_

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

#include "inference_backend.hpp"

namespace autoware_pov::AutoSeg::SceneSeg
{

class SceneSeg
{
public:
  explicit SceneSeg(const std::string & model_path, const std::string & backend, const std::string & precision, int gpu_id);

  bool doInference(const cv::Mat & input_image);
  void getRawMask(cv::Mat & raw_mask, const cv::Size & output_size) const;
  void colorizeMask(const cv::Mat & raw_mask, const cv::Mat & original_image, cv::Mat & blended_image) const;

private:
  void createColorMap();
  
  std::unique_ptr<InferenceBackend> backend_;

  // Segmentation color map for visualization
  std::vector<cv::Vec3b> color_map_;
};

}  // namespace autoware_pov::AutoSeg::SceneSeg

#endif  // AUTOWARE_POV__AUTOSEG__SCENE_SEG__SCENE_SEG_HPP_ 