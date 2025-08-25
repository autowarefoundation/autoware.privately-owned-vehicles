#ifndef MASKS_VISUALIZATION_ENGINE_HPP_
#define MASKS_VISUALIZATION_ENGINE_HPP_

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

#ifdef CUDA_FOUND
#include "cuda_visualization_kernels.hpp"
#endif

namespace autoware_pov::common {

class MasksVisualizationEngine {
public:
    explicit MasksVisualizationEngine(const std::string& viz_type, bool show_opencv_window = true);
    
    // Simple mask visualization (clean approach)
    cv::Mat visualize(const cv::Mat& mask, const cv::Mat& original_image);

private:
    
    std::vector<cv::Vec3b> color_map_;
    std::string viz_type_;
    bool use_cuda_;
    bool show_opencv_window_;  // Flag to control OpenCV window display
};

} // namespace autoware_pov::common

#endif // MASKS_VISUALIZATION_ENGINE_HPP_