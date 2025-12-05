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

class LaneTracker {
public:
    LaneTracker();
    ~LaneTracker() = default;

    /**
     * @brief Main processing func
     * 1. Warp valid egoline to BEV
     * 2. Recover missing egoline in BEV via last-known-good-frame's lane width shifting
     * 3. Calc curve params in BEV
     * 4. Warp back to perspective to update vis
     * * @param input_lanes Input from LaneFilter
     * @param image_size Size of full input img
     * @return Updated LaneSegmentation with recovered egolines and 6 curve params (3 for each view)
     */
    std::pair<LaneSegmentation, DualViewMetrics> update(
        const LaneSegmentation& input_lanes,
        const cv::Size& image_size
    );

private:
    // --- State params ---
    cv::Mat H_orig_to_bev;      // Homomatrix
    cv::Mat H_bev_to_orig;      // Inversed homomatrix
    bool homography_inited = false;
    cv::Size cached_image_size;

    // BEV lane width cache (in BEV pixels)
    // Updated whenever there are 2 valid egolines.
    // If one missing, use this to shift the available one.
    // If both lost, fuck it I'm out.
    double last_valid_bev_width = 180.0;    // Default fallback (tuned for 640x640 BEV)
    bool has_valid_width_history = false;

}