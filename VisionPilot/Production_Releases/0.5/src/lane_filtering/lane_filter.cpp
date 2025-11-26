#include "lane_filtering/lane_filter.hpp"
#include "inference/onnxruntime_engine.hpp"
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

// Master update function
LaneSegmentation LaneFilter::update(const LaneSegmentation& raw_input) {
    LaneSegmentation clean_output;
    clean_output.width = raw_input.width;   // 160
    clean_output.height = raw_input.height; // 80
    
    // Initialize blank masks
    clean_output.ego_left = cv::Mat::zeros(
        clean_output.height, 
        clean_output.width, 
        CV_32FC1
    );
    clean_output.ego_right = cv::Mat::zeros(
        clean_output.height, 
        clean_output.width, 
        CV_32FC1
    );
    clean_output.other_lanes = cv::Mat::zeros(
        clean_output.height, 
        clean_output.width, 
        CV_32FC1
    );

    