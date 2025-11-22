#include "openlane_loader.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>

using json = nlohmann::json;

OpenLaneLoader::OpenLaneLoader() {}

std::vector<FrameAnnotation> OpenLaneLoader::loadFromFolder(const std::string& jsonFolderPath, 
                                                              const std::string& imageFolderPath,
                                                              const std::string& laneFolderPath) {
    std::vector<FrameAnnotation> annotations;

    if (!fs::exists(jsonFolderPath)) {
        std::cerr << "JSON folder not found: " << jsonFolderPath << std::endl;
        return annotations;
    }

    std::vector<std::string> jsonFiles;
    for (const auto& entry : fs::recursive_directory_iterator(jsonFolderPath)) {
        if (entry.is_regular_file() && entry.path().extension() == ".json") {
            jsonFiles.push_back(entry.path().string());
        }
    }

    std::sort(jsonFiles.begin(), jsonFiles.end());

    size_t maxFiles = std::min(jsonFiles.size(), size_t(500));
    std::cout << "Loading " << maxFiles << " of " << jsonFiles.size() << " annotation files..." << std::endl;

    for (size_t i = 0; i < maxFiles; ++i) {
        const auto& jsonFile = jsonFiles[i];
        try {
            FrameAnnotation annotation = loadAnnotation(jsonFile);

            // Build image path from raw_file_path
            if (!annotation.imagePath.empty() && fs::exists(imageFolderPath)) {
                // Extract just the filename from raw_file_path
                fs::path rawPath(annotation.imagePath);
                std::string filename = rawPath.filename().string();
                
                // Construct full image path
                fs::path fullImagePath = fs::path(imageFolderPath) / filename;
                annotation.imagePath = fullImagePath.string();
            }

            // Load lane annotations if lane folder provided
            if (!laneFolderPath.empty() && fs::exists(laneFolderPath)) {
                // Extract base name (timestamp) from JSON
                fs::path jsonPath(jsonFile);
                std::string basename = jsonPath.stem().string(); // e.g., "150923227159162900.jpg"
                
                // Remove .jpg extension from basename if present
                if (basename.size() > 4 && basename.substr(basename.size() - 4) == ".jpg") {
                    basename = basename.substr(0, basename.size() - 4);
                }
                
                // Search for matching lane JSON in lane folder
                std::string laneJsonFilename = basename + ".json";
                fs::path laneJsonPath;
                
                // Search recursively in lane folder
                for (const auto& entry : fs::recursive_directory_iterator(laneFolderPath)) {
                    if (entry.is_regular_file() && entry.path().filename() == laneJsonFilename) {
                        laneJsonPath = entry.path();
                        break;
                    }
                }
                
                // Load lane data if found
                if (!laneJsonPath.empty() && fs::exists(laneJsonPath)) {
                    try {
                        std::ifstream laneFile(laneJsonPath);
                        json laneJson;
                        laneFile >> laneJson;
                        
                        if (laneJson.contains("lane_lines") && laneJson["lane_lines"].is_array()) {
                            for (const auto& laneData : laneJson["lane_lines"]) {
                                LaneLine lane;
                                lane.attribute = laneData.value("attribute", 0);
                                lane.trackId = laneData.value("track_id", 0);
                                
                                if (laneData.contains("uv") && laneData["uv"].is_array()) {
                                    const auto& uvArray = laneData["uv"];
                                    if (uvArray.size() == 2 && uvArray[0].is_array() && uvArray[1].is_array()) {
                                        const auto& uCoords = uvArray[0];
                                        const auto& vCoords = uvArray[1];
                                        
                                        size_t numPoints = std::min(uCoords.size(), vCoords.size());
                                        lane.uvPoints.reserve(numPoints);
                                        
                                        for (size_t j = 0; j < numPoints; ++j) {
                                            float u = uCoords[j].get<float>();
                                            float v = vCoords[j].get<float>();
                                            lane.uvPoints.push_back(cv::Point2f(u, v));
                                        }
                                    }
                                }
                                
                                if (!lane.uvPoints.empty()) {
                                    annotation.laneLines.push_back(std::move(lane));
                                }
                            }
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "Error loading lane annotation " << laneJsonPath << ": " << e.what() << std::endl;
                    }
                }
            }

            annotations.push_back(std::move(annotation));
        } catch (const std::exception& e) {
            std::cerr << "Error loading annotation " << jsonFile << ": " << e.what() << std::endl;
        }
    }

    std::cout << "Loaded " << annotations.size() << " annotations" << std::endl;
    return annotations;
}

FrameAnnotation OpenLaneLoader::loadAnnotation(const std::string& jsonFilePath) {
    FrameAnnotation result;

    std::ifstream file(jsonFilePath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + jsonFilePath);
    }

    json j;
    file >> j;

    // Expect new format: raw_file_path and result[]
    if (j.contains("raw_file_path") && j["raw_file_path"].is_string()) {
        result.imagePath = j["raw_file_path"].get<std::string>();
    }

    if (j.contains("result") && j["result"].is_array()) {
        for (const auto& boxData : j["result"]) {
            DetectionBox box;
            box.x = boxData.value("x", 0.0);
            box.y = boxData.value("y", 0.0);
            box.width = boxData.value("width", 0.0);
            box.height = boxData.value("height", 0.0);
            box.type = boxData.value("type", 0);
            box.id = boxData.value("id", 0);
            
            // trackid can be string or number
            if (boxData.contains("trackid")) {
                if (boxData["trackid"].is_string()) {
                    box.trackid = boxData["trackid"].get<std::string>();
                } else if (boxData["trackid"].is_number()) {
                    box.trackid = std::to_string(boxData["trackid"].get<int>());
                }
            }
            
            result.detectionBoxes.push_back(box);
        }
    }

    return result;
}
