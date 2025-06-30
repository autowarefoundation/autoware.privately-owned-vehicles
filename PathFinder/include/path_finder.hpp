#ifndef PATH_FINDER_HPP
#define PATH_FINDER_HPP

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <filesystem>

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

void drawLanes(const std::vector<LanePts> &lanes,
               const std::vector<fittedCurve> &egoLanes,
               const fittedCurve &egoPath);
std::vector<LanePts> loadLanesFromYaml(const std::string &filename);
std::array<double, 2> generatePixelNoise(double max_noise);
std::array<double, 3> fitQuadPoly(const std::vector<cv::Point2f> &points);
cv::Mat loadHFromYaml(const std::string &filename);
void cameraView(const std::vector<std::vector<cv::Point2f>> &lanes2d);
fittedCurve calculateEgoPath(const fittedCurve &leftLane, const fittedCurve &rightLane);
void estimateH();

class Estimator
{
private:
    size_t dim;                   // dimensionality
    std::vector<double> state;    // mean estimate for each variable
    std::vector<double> variance; // variance for each variable (diagonal covariance)
public:
    void initialize(const std::vector<double> &init_state, const std::vector<double> &init_var);
    void predict(std::vector<double> process_var); // Add process noise to variance
    void update(const std::vector<double> &measurement, const std::vector<double> &measurement_var);
    const std::vector<double> &getState() const;
    const std::vector<double> &getVariance() const;
};

#endif // PATH_FINDER_HPP