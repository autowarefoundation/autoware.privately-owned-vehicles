#include "lane_filtering/lane_filter.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace autoware_pov::vision::autosteer
{

LaneFilter::LaneFilter(float smoothing_factor) : alpha(smoothing_factor) {}

void LaneFilter::reset() {
    prev_left_fit.valid = false;
    prev_right_fit.valid = false;
}