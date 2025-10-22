#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <filesystem>
#include <optional>
#include "estimator.hpp"

struct LanePts
{
    int id;
    std::vector<cv::Point2f> BevPoints; // meters
    std::vector<cv::Point2f> GtPoints;  // meters
    LanePts(int id,
            std::vector<cv::Point2f> GtPoints,
            std::vector<cv::Point2f> BevPoints);
};

struct fittedCurve
{
    std::array<double, 3> coeff; // Coefficients for the quadratic polynomial
    double cte;                  // Cross-track error in meters
    double yaw_error;            // Yaw error in radians
    double curvature;            // Curvature in meters^-1
    fittedCurve(const std::array<double, 3> &coeff);
};

struct drivingCorridor{
    fittedCurve egoLaneL;
    fittedCurve egoLaneR;
    fittedCurve egoPath;
    double cte;                  // Cross-track error in meters
    double yaw_error;            // Yaw error in radians
    double curvature;            // Curvature in meters^-1
    double width;
    drivingCorridor(
        const fittedCurve &left,
        const fittedCurve &right,
        const fittedCurve &path);
};

std::array<double, 3> fitQuadPoly(const std::vector<cv::Point2f> &points);
cv::Mat loadHFromYaml(const std::string &filename);
fittedCurve calculateEgoPath(const fittedCurve &leftLane, const fittedCurve &rightLane);
void estimateH(const std::string &filename);