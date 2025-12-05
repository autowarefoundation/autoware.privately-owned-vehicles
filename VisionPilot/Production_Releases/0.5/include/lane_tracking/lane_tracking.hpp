#include "inference/onnxruntime_engine.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <array>

namespace autoware_pov::vision::autosteer
{

/**
 * @brief Container for curve params both views
 * (for statistics and debugging later)
 */
struct DualViewMetrics {

    // Original perspective curve params
    double pers_lane_offset = 0.0;
    double pers_yaw_offset = 0.0;
    double pers_curvature = 0.0;

    // BEV perspective curve params
    double bev_lane_offset = 0.0;
    double bev_yaw_offset = 0.0;
    double bev_curvature = 0.0;
};

}