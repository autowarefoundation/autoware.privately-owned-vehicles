#include "annotation_visualizer.h"
#include <opencv2/imgproc.hpp>

AnnotationVisualizer::AnnotationVisualizer() {}

cv::Mat AnnotationVisualizer::renderAnnotations(const cv::Mat& image, const FrameAnnotation& annotation) {
    cv::Mat result = image.clone();
    
    drawLanes(result, annotation);
    drawDetections(result, annotation);
    
    return result;
}

void AnnotationVisualizer::drawLanes(cv::Mat& image, const FrameAnnotation& annotation) {
    // Draw lane lines using UV coordinates
    for (const auto& lane : annotation.laneLines) {
        if (lane.uvPoints.empty()) {
            continue;
        }
        
        // Choose color based on attribute (1=left-left, 2=ego-left, 3=ego-right, 4=right-right)
        cv::Scalar laneColor;
        
        if (lane.attribute == 2) {
            laneColor = colorEgoLeftLane;
        } else if (lane.attribute == 3) {
            laneColor = colorEgoRightLane;
        } else {
            laneColor = colorOtherLanes;
        }
        
        // Convert points to integer points for polylines
        std::vector<cv::Point> intPoints;
        intPoints.reserve(lane.uvPoints.size());
        for (const auto& pt : lane.uvPoints) {
            int x = static_cast<int>(std::round(pt.x));
            int y = static_cast<int>(std::round(pt.y));
            
            // Clamp to image boundaries
            x = std::max(0, std::min(x, image.cols - 1));
            y = std::max(0, std::min(y, image.rows - 1));
            
            intPoints.push_back(cv::Point(x, y));
        }
        
        // Draw polyline with thickness 3
        if (intPoints.size() >= 2) {
            cv::polylines(image, intPoints, false, laneColor, 3, cv::LINE_AA);
        }
    }
}

void AnnotationVisualizer::drawDetections(cv::Mat& image, const FrameAnnotation& annotation) {
    if (annotation.detectionBoxes.empty()) {
        return;
    }

    // Find closest car (highest y-coordinate bottom, assuming camera perspective)
    size_t closestIndex = 0;
    float maxBottomY = 0;
    for (size_t i = 0; i < annotation.detectionBoxes.size(); ++i) {
        const auto& box = annotation.detectionBoxes[i];
        float bottomY = static_cast<float>(box.y + box.height);
        if (bottomY > maxBottomY) {
            maxBottomY = bottomY;
            closestIndex = i;
        }
    }

    // Draw detection boxes
    for (size_t i = 0; i < annotation.detectionBoxes.size(); ++i) {
        const auto& box = annotation.detectionBoxes[i];
        float x = static_cast<float>(box.x);
        float y = static_cast<float>(box.y);
        float w = static_cast<float>(box.width);
        float h = static_cast<float>(box.height);

        cv::Point2f tl(x, y);
        cv::Point2f br(x + w, y + h);

        // Clamp to image
        if (tl.x < 0) tl.x = 0;
        if (tl.y < 0) tl.y = 0;
        if (br.x >= image.cols) br.x = image.cols - 1;
        if (br.y >= image.rows) br.y = image.rows - 1;

        if (tl.x < br.x && tl.y < br.y) {
            bool isClosest = (i == closestIndex);
            cv::Scalar boxColor = isClosest ? colorClosestCar : colorOtherCars;
            
            if (isClosest) {
                // Fill closest car with transparency
                cv::Mat overlay = image.clone();
                cv::rectangle(overlay, tl, br, boxColor, -1);  // Filled rectangle
                cv::addWeighted(image, 0.7, overlay, 0.3, 0, image);  // 30% opacity
                // Draw border
                cv::rectangle(image, tl, br, boxColor, 3);
                
                // Add speed and distance text below the ego car box
                float centerX = (tl.x + br.x) / 2.0f;
                float textY = br.y + 25;  // Position text below the box
                
                // Dummy values for speed and distance
                std::string speedText = "Speed: 20 mph";
                std::string distanceText = "Distance: 10 ft";
                
                // Draw speed text
                cv::Size speedTextSize = cv::getTextSize(speedText, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr);
                cv::Point speedTextPos(static_cast<int>(centerX - speedTextSize.width / 2), static_cast<int>(textY));
                cv::putText(image, speedText, speedTextPos, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
                cv::putText(image, speedText, speedTextPos, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
                
                // Draw distance text
                cv::Size distanceTextSize = cv::getTextSize(distanceText, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr);
                cv::Point distanceTextPos(static_cast<int>(centerX - distanceTextSize.width / 2), static_cast<int>(textY + 25));
                cv::putText(image, distanceText, distanceTextPos, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
                cv::putText(image, distanceText, distanceTextPos, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
                
            } else {
                // Draw outline only for other cars
                cv::rectangle(image, tl, br, boxColor, 2);
            }
        }
    }
}
