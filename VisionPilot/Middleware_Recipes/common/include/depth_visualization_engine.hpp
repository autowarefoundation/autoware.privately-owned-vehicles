#ifndef DEPTH_VISUALIZATION_ENGINE_HPP_
#define DEPTH_VISUALIZATION_ENGINE_HPP_

#include <opencv2/opencv.hpp>

namespace autoware_pov::common {

class DepthVisualizationEngine {
public:
    DepthVisualizationEngine();
    
    // Core depth visualization logic - framework agnostic
    cv::Mat visualize(const cv::Mat& depth_map);
    
private:
    // No parameters needed - depth visualization is standardized
};

} // namespace autoware_pov::common

#endif // DEPTH_VISUALIZATION_ENGINE_HPP_