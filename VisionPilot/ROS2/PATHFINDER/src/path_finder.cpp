#include "path_finder.hpp"

LanePts::LanePts(int id,
                 std::vector<cv::Point2f> GtPoints,
                 std::vector<cv::Point2f> BevPoints) : id(id), GtPoints(GtPoints), BevPoints(BevPoints) {}

fittedCurve::fittedCurve(const std::array<double, 3> &coeff) : coeff(coeff)
{
    cte = -coeff[2];
    yaw_error = -atan2(coeff[1], 1.0);
    curvature = 2 * coeff[0] / std::pow(1 + coeff[1] * coeff[1], 1.5);
}

drivingCorridor::drivingCorridor(
                 const fittedCurve &left,
                 const fittedCurve &right,
                 const fittedCurve &path)
    : egoLaneL(left), egoLaneR(right), egoPath(path), width(4.0)
{
    //TODO: deal with missing lanes here
    // if (egoLaneL && egoLaneR)
    // {
    //     if (!egoPath)
    //     {
            const auto &cl = egoLaneL.coeff;
            const auto &cr = egoLaneR.coeff;
            egoPath = fittedCurve({
                            (cl[0] + cr[0]) / 2.0,
                            (cl[1] + cr[1]) / 2.0,
                            (cl[2] + cr[2]) / 2.0
                        });
        // }
        width = egoLaneL.cte - egoLaneR.cte;
        std::cout << "corridor width is " << width << std::endl;
//     }
}

cv::Mat loadHFromYaml(const std::string &filename)
{
    YAML::Node root = YAML::LoadFile(filename);
    const auto &mat = root["H"];

    if (!mat || mat.size() != 9)
    {
        throw std::runtime_error("Invalid or missing homography matrix (expecting 9 values).");
    }

    cv::Mat H = cv::Mat(3, 3, CV_64F); // Create 3x3 matrix of type double

    for (int i = 0; i < 9; ++i)
    {
        H.at<double>(i / 3, i % 3) = mat[i].as<double>();
    }

    // std::cout << "Loaded H:\n"
    //           << H << std::endl;
    return H;
}

std::array<double, 3> fitQuadPoly(const std::vector<cv::Point2f> &points)
{
    const int degree = 2;
    const size_t N = points.size();

    if (points.size() == 2)
    {
        double y1 = points[0].y, x1 = points[0].x;
        double y2 = points[1].y, x2 = points[1].x;

        double b = (x2 - x1) / (y2 - y1);
        double c = x1 - b * y1;

        return {0.0, b, c};
    }

    Eigen::MatrixXd A(N, degree + 1);
    Eigen::VectorXd b(N);

    for (size_t i = 0; i < N; ++i)
    {
        double x = points[i].x;
        double y = points[i].y;

        A(i, 0) = y * y;
        A(i, 1) = y;
        A(i, 2) = 1.0;
        b(i) = x;
    }
    std::array<double, 3> coeffs;
    Eigen::VectorXd res = A.colPivHouseholderQr().solve(b);
    for (int i = 0; i <= degree; ++i)
    {
        coeffs[i] = res(i);
    }
    return coeffs;
}

void estimateH(const std::string &filename)
{
    //   [975.577, 567.689]    //   [1.657, 1.973, 38.649]
    //   [1150.05, 747.249]    //   [1.469, 1.553, 6.53]
    //   [904.635, 567.044]    //   [-1.217, 1.968, 39.064]
    //   [706.126, 741.3]      //   [-1.583, 1.539, 6.644]

    std::vector<cv::Point2f> imagePixels = {
        {1150.05f, 747.249f},
        {706.126f, 741.3f},
        {904.635f, 567.044f},
        {975.577f, 567.689f}};
    std::vector<cv::Point2f> BevPixels = {
        {1.469f, 6.53f},
        {-1.583f, 6.644f},
        {-1.217f, 39.064f},
        {1.657f, 38.649f}};
    cv::Mat H = cv::findHomography(imagePixels, BevPixels);
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    file << "H: " << H.reshape(1, 1) << std::endl;
}

fittedCurve calculateEgoPath(const fittedCurve &leftLane, const fittedCurve &rightLane)
{
    return fittedCurve({(leftLane.coeff[0] + rightLane.coeff[0]) / 2.0,
                        (leftLane.coeff[1] + rightLane.coeff[1]) / 2.0,
                        (leftLane.coeff[2] + rightLane.coeff[2]) / 2.0});
}