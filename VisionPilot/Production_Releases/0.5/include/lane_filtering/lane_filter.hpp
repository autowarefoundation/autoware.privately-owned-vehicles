#include "inference/onnxruntime_engine.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <random>

namespace autoware_pov::vision::autosteer
{

/**
 * @brief Struct to hold lane polynomial fitting results
 */
struct LanePolyFit {
    std::vector<double> coeffs;  // Cubic polyfit coeffs [a, b, c, d] for ax^3 + bx^2 + cx + d
    bool valid;                  // Whether the lane fit is valid
};

class LaneFilter {
public:
    explicit LaneFilter(float smoothing_factor = 0.5f);

    /**
        * @brief Main processng func
        * ROI starting points => Sliding window => Polyfit => Render clean masks
    */
    LaneSegmentation update(const LaneSegmentation& raw_input);

    void reset();

private:
    const int roi_y_min = 40;   // Start scanning from this of 80px mask height
    const int roi_y_max = 79;  // End scanning at bottom of mask
    const int sliding_window_height = 4;
    const int sliding_window_width = 8;
    const int min_pixels_for_fit = 5;
    const int consecutive_empty_threshold = 12;

    // RANSAC polyfit params
    const int ransac_iterations = 50;       // Combi of iterations
    const double ransac_threshold = 2.0;    // Max reproj error
    std::mt19937 rng;                       // Random number generator

    // History holder for lane recovery strategy
    LanePolyFit last_strong_left;
    LanePolyFit last_strong_right;
    double last_lane_width_bottom = -1.0;
    bool has_strong_history = false;

    // RANSAC helper func
    std::vector<double> fitPolySimple(
        const std::vector<cv::Point>& subset, 
        int order
    );

    // Error calc func
    double getError(
        const std::vector<double>& coeffs, 
        const cv::Point& p
    );

    // Helper funcs

    // Step 1: ROI for starting points
    void findStartingPoints(
        const LaneSegmentation& raw,
        std::vector<int>& start_left,
        std::vector<int>& start_right
    );

    // Step 2: sliding window search
    std::vector<cv::Point> slidingWindowSearch(
        const LaneSegmentation& raw,
        cv::Point start_point,
        bool is_left_lane
    );

    // Step 3: cubic polynomial fit
    LanePolyFit fitPoly(const std::vector<cv::Point>& points);

    // Evaluate polyfitted line helper
    double evalPoly(const std::vector<double>& coeffs, double y);

    // State for temporal smoothing
    LanePolyFit prev_left_fit;
    LanePolyFit prev_right_fit;
    float smoothing_factor;
};

}  // namespace autoware_pov::vision::autosteer