#pragma once

#include <opencv2/core.hpp>
#include "../io/openlane_loader.h"

class AnnotationVisualizer {
public:
    AnnotationVisualizer();
    
    // Render annotations on an image
    // Returns a copy of the image with annotations drawn
    cv::Mat renderAnnotations(const cv::Mat& image, const FrameAnnotation& annotation);
    
private:
    // Draw lane lines (UV coordinates)
    void drawLanes(cv::Mat& image, const FrameAnnotation& annotation);

    // Draw detection boxes
    void drawDetections(cv::Mat& image, const FrameAnnotation& annotation);
    
    // Color constants
    cv::Scalar colorEgoLeftLane = cv::Scalar(255, 0, 0);     // Blue for ego-left lane
    cv::Scalar colorEgoRightLane = cv::Scalar(255, 0, 255);  // Pink for ego-right lane
    cv::Scalar colorOtherLanes = cv::Scalar(0, 255, 0);      // Green for other lanes
    cv::Scalar colorOtherCars = cv::Scalar(255, 0, 0);       // Blue for other cars
    cv::Scalar colorClosestCar = cv::Scalar(0, 0, 255);      // Red for closest car
};
