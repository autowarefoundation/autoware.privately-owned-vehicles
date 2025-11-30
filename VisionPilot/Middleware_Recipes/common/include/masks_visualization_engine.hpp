#ifndef MASKS_VISUALIZATION_ENGINE_HPP_
#define MASKS_VISUALIZATION_ENGINE_HPP_

#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>

namespace autoware_pov::common {

class MasksVisualizationEngine {
public:
    explicit MasksVisualizationEngine(const std::string& viz_type, bool show_opencv_window = true);
    
    /**
     * Visualize mask by creating colored overlay and blending with original image
     * @param mask Input binary mask (MONO8)
     * @param original_image Original image to blend with
     * @param latency_ms ROS pipeline latency in milliseconds
     * @param fps ROS pipeline FPS
     * @return Blended result image
     */
    cv::Mat visualize(const cv::Mat& mask, const cv::Mat& original_image, 
                     double latency_ms = -1.0, double fps = -1.0);

private:
    /**
     * Create colored mask based on visualization type
     * @param mask Input binary mask
     * @return Colored mask (BGR8)
     */
    cv::Mat createColorMask(const cv::Mat& mask);
    
    /**
     * Display OpenCV window with performance metrics overlay
     * @param image Image to display
     * @param latency_ms ROS latency
     * @param fps ROS FPS
     * @param colorize_time_ms Colorization time
     */
    void displayOpenCVWindow(const cv::Mat& image, double latency_ms, double fps, double colorize_time_ms);
    
    // Configuration
    std::string viz_type_;
    bool show_opencv_window_;
    
    // OpenCV FPS measurement
    std::chrono::steady_clock::time_point last_opencv_time_;
    int opencv_frame_count_;
    double opencv_fps_;
};

} // namespace autoware_pov::common

#endif // MASKS_VISUALIZATION_ENGINE_HPP_