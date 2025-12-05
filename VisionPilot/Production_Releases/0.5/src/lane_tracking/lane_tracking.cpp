// This script takes the output of the lane fitler.

// The blue and pink perspective image polynomial fitted lines need to be sampled and the samples should be transformed into a BEV space using a Homography transform (please note - we are not creating a BEV image, only transforming discrete points from the perspective coordinates to the BEV coordinates).

// To do this, you can simply multiply the coordinates of the polyfit line samples by the Homography matrix.

// Verify this is working by visualizing the transformed points.

// Once this has been verfied, we will proceed with the drivable corridor parameter estimation and temporal tracking.

#include "lane_tracking/lane_tracking.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>


namespace autoware_pov::vision::autosteer
{

LaneTracker::LaneTracker() {
    // Homography is pre-computed elsewhere and hard-coded.
    // If you got any problem with it, ask me.
}

void LaneTracker::initHomography(const cv::Size& image_size) {

    if (
        homography_inited && 
        cached_image_size == image_size
    ) return;

    cached_image_size = image_size;
    homography_inited = true;
    
}

