#include "include/inference/onnxruntime_engine.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>

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
    const int roi_y_min = 50;   // Start scanning from this of 80px mask height
    const int roi_y_max = 79;  // End scanning at bottom of mask
    const int sliding_window_height = 5;
    const int sliding_window_width = 16;
    const int min_pixels_for_fit = 5;

    