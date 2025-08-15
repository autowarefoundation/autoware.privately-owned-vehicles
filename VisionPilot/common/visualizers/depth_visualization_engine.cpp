#include "../include/depth_visualization_engine.hpp"

namespace autoware_pov::common {

DepthVisualizationEngine::DepthVisualizationEngine() {
    // No initialization needed for depth visualization
}

cv::Mat DepthVisualizationEngine::visualize(const cv::Mat& depth_map) {
    // Normalize depth to 0-255 range (extracted from original visualize_depth_node.cpp)
    cv::Mat normalized_depth;
    double min_val, max_val;
    cv::minMaxLoc(depth_map, &min_val, &max_val);
    
    if (max_val > min_val) {
        depth_map.convertTo(normalized_depth, CV_8UC1, 255.0 / (max_val - min_val), -min_val * 255.0 / (max_val - min_val));
    } else {
        normalized_depth = cv::Mat::zeros(depth_map.size(), CV_8UC1);
    }
    
    // Apply colormap (VIRIDIS for depth visualization)
    cv::Mat colorized_depth;
    cv::applyColorMap(normalized_depth, colorized_depth, cv::COLORMAP_VIRIDIS);
    
    return colorized_depth;
}

} // namespace autoware_pov::common