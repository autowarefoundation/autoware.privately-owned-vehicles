#ifndef MASKS_VISUALIZATION_ENGINE_HPP_
#define MASKS_VISUALIZATION_ENGINE_HPP_

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

namespace autoware_pov::common {

class MasksVisualizationEngine {
public:
    explicit MasksVisualizationEngine(const std::string& viz_type);
    
    // Core visualization logic - framework agnostic
    cv::Mat visualize(const cv::Mat& mask, const cv::Mat& original_image);
    
private:
    void createColorMap(const std::string& viz_type);
    
    std::vector<cv::Vec3b> color_map_;
    std::string viz_type_;
};

} // namespace autoware_pov::common

#endif // MASKS_VISUALIZATION_ENGINE_HPP_