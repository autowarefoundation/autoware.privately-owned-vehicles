#pragma once

#include <string>
#include <vector>
#include <filesystem>
#include <opencv2/core.hpp>

namespace fs = std::filesystem;

// Object detection bounding box
struct DetectionBox {
    double x;
    double y;
    double width;
    double height;
    int type;
    int id;
    std::string trackid;
};

// Lane line with 2D UV coordinates
struct LaneLine {
    std::vector<cv::Point2f> uvPoints;  // 2D image coordinates
    int attribute;  // Lane attribute (0=centerline, 1=left, 2=right, etc.)
    int trackId;
};

// Annotation data for a frame
struct FrameAnnotation {
    std::string imagePath;  // Path to image
    std::vector<DetectionBox> detectionBoxes;  // Detection boxes
    std::vector<LaneLine> laneLines;  // Lane lines
};

// Main loader class
class OpenLaneLoader {
public:
    OpenLaneLoader();
    
    // Load annotations from a specified folder with images
    std::vector<FrameAnnotation> loadFromFolder(const std::string& jsonFolderPath, 
                                                 const std::string& imageFolderPath,
                                                 const std::string& laneFolderPath = "");
    
    // Load a single annotation file
    FrameAnnotation loadAnnotation(const std::string& jsonFilePath);
};
