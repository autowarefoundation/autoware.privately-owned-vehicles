#ifndef MASKS_VISUALIZATION_ENGINE_HPP_
#define MASKS_VISUALIZATION_ENGINE_HPP_

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

#ifdef CUDA_FOUND
#include "masks_visualization_kernels.hpp"
#endif

namespace autoware_pov::common {

class MasksVisualizationEngine {
public:
    explicit MasksVisualizationEngine(const std::string& viz_type, bool show_opencv_window = true);
    
    // Simple mask visualization (clean approach)
    cv::Mat visualize(const cv::Mat& mask, const cv::Mat& original_image, 
                     double latency_ms = -1.0, double fps = -1.0);

private:
    
    std::vector<cv::Vec3b> color_map_;
    std::string viz_type_;
    bool use_cuda_;
    bool show_opencv_window_;  // Flag to control OpenCV window display
    
    // OpenCV FPS measurement
    std::chrono::steady_clock::time_point last_opencv_time_;
    int opencv_frame_count_;
    double opencv_fps_;
};

} // namespace autoware_pov::common

#endif // MASKS_VISUALIZATION_ENGINE_HPP_