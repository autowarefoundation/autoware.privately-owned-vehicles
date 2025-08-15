#include "path_finder.hpp"

// TODO: fuse yaw_error and curv using product of Gaussians
bool gt = true; // true for ground truth, false for BEV points

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
    const std::optional<fittedCurve> &left,
    const std::optional<fittedCurve> &right,
    const std::optional<fittedCurve> &path)
    : egoLaneL(left), egoLaneR(right), egoPath(path)
{
    if (egoLaneL && egoLaneR)
    {
        if (!egoPath)
        {
            egoPath = fittedCurve({(egoLaneL->coeff[0] + egoLaneR->coeff[0]) / 2.0,
                                   (egoLaneL->coeff[1] + egoLaneR->coeff[1]) / 2.0,
                                   (egoLaneL->coeff[2] + egoLaneR->coeff[2]) / 2.0});
        }
        width = egoLaneL->cte - egoLaneR->cte;
        std::cout << "corridor width is " << width << std::endl;
    }
}

void drawLanes(const std::vector<LanePts> &lanes,
               const std::vector<fittedCurve> &egoLanes,
               const fittedCurve &egoPath)
{
    int width = 800, height = 1000, scale = 20, height_offset = 50;
    cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);
    cv::line(image, cv::Point(width / 2, 0), cv::Point(width / 2, height), cv::Scalar(255, 255, 255), 1); // white centerline
    cv::circle(image, cv::Point(width / 2, height - height_offset), 5, cv::Scalar(0, 0, 255), -1);        // red dot

    int color = 0;
    for (const auto &lane : lanes)
    {
        int prev_u, prev_v;
        color += 20;
        std::vector<cv::Point2f> points;
        if (gt)
            points = lane.GtPoints;
        else
            points = lane.BevPoints;

        for (int i = 0; i < points.size(); i++)
        {
            const auto pt_m = points[i];
            int u = pt_m.x * scale + width / 2;
            int v = height - height_offset - pt_m.y * scale;
            cv::Point pt_pix(u, v);
            cv::circle(image, pt_pix, 3, cv::Scalar(255 - color, color, 0), -1); // green dots
            if (i > 0)
            {
                cv::Point prev(prev_u, prev_v);
                cv::line(image, prev, pt_pix, cv::Scalar(0, 255, 0), 1); // green line
            }
            prev_u = u;
            prev_v = v;
        }
    }

    color = 0;
    for (auto egoLane : egoLanes)
    {
        cv::Point2f prev_pt(0, 0);
        for (double y = 0.0; y <= 50.0; y += 0.1) // y range in meters
        {
            double x = egoLane.coeff[0] * y * y + egoLane.coeff[1] * y + egoLane.coeff[2];
            int u = static_cast<int>(x * scale + width / 2);
            int v = static_cast<int>(height - height_offset - y * scale);
            if (y > 0.0) // skip the first point
            {
                cv::line(image, prev_pt, cv::Point2f(u, v), cv::Scalar(0, color, 255), 1); // yellow
            }
            prev_pt.x = u;
            prev_pt.y = v;
        }
        color += 100;
    }

    // Draw ego path
    cv::Point2f prev_pt(0, 0);
    for (double y = 0.0; y <= 50.0; y += 0.1) // y range in meters
    {
        double x = egoPath.coeff[0] * y * y + egoPath.coeff[1] * y + egoPath.coeff[2];
        int u = static_cast<int>(x * scale + width / 2);
        int v = static_cast<int>(height - height_offset - y * scale);
        if (y > 0.0) // skip the first point
        {
            cv::line(image, prev_pt, cv::Point2f(u, v), cv::Scalar(255, 0, 0), 2);
        }
        prev_pt.x = u;
        prev_pt.y = v;
    }

    cv::imshow("BEV Lanes", image);
    // cv::imwrite("../test/bev_lanes.png", image);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

std::vector<LanePts> loadLanesFromYaml(const std::string &filename, cv::Mat &H)
{
    std::vector<LanePts> lanes;
    YAML::Node root = YAML::LoadFile(filename);
    int i = 0;
    for (const auto &lane3d : root["lanes3d"])
    {
        const auto &lane2d = root["lanes2d"][i++];
        // std::cout << "LanePts " << i << ":\n";
        std::vector<cv::Point2f> lane_pixels;
        for (const auto &pt2d : lane2d)
        {
            // std::cout << "  [" << pt2d[0].as<double>() << ", " << pt2d[1].as<double>() << "]\n";
            auto noise = generatePixelNoise(5.0);
            double u = pt2d[0].as<double>() ;//+ noise[0];
            double v = pt2d[1].as<double>() ;//+ noise[1];
            lane_pixels.emplace_back(cv::Point2f(u, v));
        }
        std::vector<cv::Point2f> bev_pixels;
        cv::perspectiveTransform(lane_pixels, bev_pixels, H);

        std::vector<cv::Point2f> gt_pts;
        for (const auto &pt3d : lane3d)
        {
            // std::cout << "  [" << pt3d[0].as<double>() << ", "
            //           << pt3d[1].as<double>() << ", "
            //           << pt3d[2].as<double>() << "]\n";
            gt_pts.emplace_back(cv::Point2f(pt3d[0].as<double>(), pt3d[2].as<double>()));
        }
        lanes.emplace_back(i, gt_pts, bev_pixels);
    }
    return lanes;
}

std::array<double, 2> generatePixelNoise(double max_noise = 10.0)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist_pix(-max_noise, max_noise);
    double du = dist_pix(gen);
    double dv = dist_pix(gen);
    return {du, dv}; // Pixel-space noise to be added to (u, v)
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

void cameraView(const std::vector<std::vector<cv::Point2f>> &lanes2d) // camera perspective
{
    // Create blank image
    int width = 1920, height = 1020;
    cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);

    // Draw each lane
    for (const auto &lane : lanes2d)
    {
        for (size_t i = 0; i < lane.size(); ++i)
        {
            cv::circle(image, lane[i], 3, cv::Scalar(0, 255, 0), -1); // green dots

            // Optionally connect points with a line
            if (i > 0)
            {
                cv::line(image, lane[i - 1], lane[i], cv::Scalar(255, 0, 0), 1); // blue line
            }
        }
    }
    cv::imshow("Camera View", image);
    cv::waitKey(0);
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

const std::vector<std::pair<int, int>> lanePairs = { // manually label LR egoLanes
    {2, 1},
    {2, 0},
    {2, 0},
    {1, 0},
    {0, 2},
    {2, 0},
    {2, 1},
    {3, 0}};

// int main()
// {
//     namespace fs = std::filesystem;
//     std::vector<std::pair<std::string, std::string>> file_pairs;
//     for (const auto &entry : fs::directory_iterator("../test/000001"))
//     {
//         if (entry.is_regular_file() && entry.path().extension() == ".yaml")
//         {
//             std::string stem = entry.path().stem().string();
//             std::string yaml_path = entry.path().string();
//             std::string img_path = "../test/000001/img/" + stem + ".jpg";

//             std::cout << "Found YAML file: " << stem << std::endl;
//             file_pairs.emplace_back(yaml_path, img_path);
//         }
//     }

//     // Sort numerically by ID (stem of filename)
//     std::sort(file_pairs.begin(), file_pairs.end(),
//               [](const auto &a, const auto &b)
//               {
//                   return std::stoll(fs::path(a.first).stem().string()) <
//                          std::stoll(fs::path(b.first).stem().string());
//               });

//     std::vector<fittedCurve> egoPaths, egoPathsGT;
//     std::vector<drivingCorridor> drivCorr, drivCorrGT;

//     int i = 0;
//     std::cout << "Found " << file_pairs.size() << " YAML files." << std::endl;
//     for (const auto &file_pair : file_pairs)
//     {
//         auto egoLanesPts = loadLanesFromYaml(file_pair.first);
//         if (egoLanesPts.size() < 2)
//         {
//             std::cerr << "Not enough lanes to calculate ego path. Loop id: " << i << std::endl;
//             continue;
//         }
//         auto egoLanesPtsLR = {egoLanesPts[lanePairs[i].first], egoLanesPts[lanePairs[i].second]};
//         i++;

//         std::vector<fittedCurve> egoLanes, egoLanesGT;

//         for (auto lanePts : egoLanesPtsLR)
//         {
//             std::array<double, 3> coeff_gt = fitQuadPoly(lanePts.GtPoints);
//             std::array<double, 3> coeff = fitQuadPoly(lanePts.BevPoints);
//             egoLanesGT.emplace_back(fittedCurve(coeff_gt));
//             egoLanes.emplace_back(fittedCurve(coeff));
//         }

//         drivCorr.emplace_back(egoLanes[0], egoLanes[1], std::nullopt);
//         drivCorrGT.emplace_back(egoLanesGT[0], egoLanesGT[1], std::nullopt);

//         cv::Mat img = cv::imread(file_pair.second, cv::IMREAD_COLOR);
//         if (img.empty())
//         {
//             std::cerr << "Failed to load image: " << file_pair.second << std::endl;
//             continue;
//         }
//         cv::imshow("Camera View", img);

//         drawLanes(egoLanesPts, egoLanes, drivCorr.back().egoPath.value());
//     }

//     // ----------------------
//     auto bayesFilter = Estimator(); // Keeps track of individual sources
//     bayesFilter.configureFusionGroups({
//         // {start_idx,end_idx}
//         // fuse indices 1,2 → result in index 3
//         {0, 3}, // cte1, cte2 → fused at index 3
//         {5, 7}, // yaw1, yaw2 → fused at index 7
//         {9, 11} // curv1, curv2 → fused at index 11
//     });

//     const double proc_SD = 0.5;
//     const double meas_SD = 0.5;
//     const double epsilon = 0.05;
//     std::random_device rd;
//     std::default_random_engine generator(rd());
//     std::uniform_real_distribution<double> dist(-epsilon, epsilon);
//     std::cout << "process standard dev: " << proc_SD << std::endl;
//     std::cout << "process mean random shift range: +-" << epsilon << std::endl;
//     std::cout << "measurement standard dev: " << meas_SD << std::endl;

//     // cte, yaw_error, curvature, driving corridor width
//     std::array<Gaussian, STATE_DIM> measurement;
//     std::array<Gaussian, STATE_DIM> process;

//     for (size_t i = 0; i < STATE_DIM; ++i)
//     {
//         process[i].mean = dist(generator);
//         process[i].variance = proc_SD * proc_SD;
//         measurement[i].variance = meas_SD * meas_SD;
//     }

//     // Main loop
//     for (int i = 0; i < drivCorr.size(); ++i)
//     {
//         const auto &egoPath = drivCorr[i].egoPath;
//         const auto &egoLaneL = drivCorr[i].egoLaneL;
//         const auto &egoLaneR = drivCorr[i].egoLaneR;
//         const auto &egoPathGT = drivCorrGT[i].egoPath;

//         measurement[0].mean = egoPath->cte;
//         measurement[1].mean = egoLaneL->cte - drivCorr[i].width / 2.0;
//         measurement[2].mean = egoLaneR->cte + drivCorr[i].width / 2.0;
//         measurement[3].mean = 0.0;

//         measurement[4].mean = egoPath->yaw_error;
//         measurement[5].mean = egoLaneL->yaw_error;
//         measurement[6].mean = egoLaneR->yaw_error;
//         measurement[7].mean = 0.0;

//         measurement[8].mean = egoPath->curvature;
//         measurement[9].mean = egoLaneL->curvature;
//         measurement[10].mean = egoLaneR->curvature;
//         measurement[11].mean = 0.0;

//         measurement[12].mean = drivCorr[i].width;

//         if (i == 0)
//         {
//             bayesFilter.initialize(measurement);
//         }
//         else
//         {
//             bayesFilter.update(measurement);
//         }

//         const auto &state = bayesFilter.getState();

//         std::cout << std::fixed << std::setprecision(4);
//         int w = 8;
//         // Header
//         std::cout << "Frame " << i << "\n";
//         std::cout << "\033[32m"
//                   << std::setw(w) << "Label"
//                   << std::setw(w) << "M_CTE" << std::setw(w) << "var"
//                   << std::setw(w) << "L_CTE" << std::setw(w) << "var"
//                   << std::setw(w) << "R_CTE" << std::setw(w) << "var"
//                   << std::setw(w) << "MLR_CTE" << std::setw(w) << "var"
//                   << std::setw(w) << "M_Yaw" << std::setw(w) << "var"
//                   << std::setw(w) << "L_Yaw" << std::setw(w) << "var"
//                   << std::setw(w) << "R_Yaw" << std::setw(w) << "var"
//                   << std::setw(w) << "LR_Yaw" << std::setw(w) << "var"
//                   << std::setw(w) << "M_Curv" << std::setw(w) << "var"
//                   << std::setw(w) << "L_Curv" << std::setw(w) << "var"
//                   << std::setw(w) << "R_Curv" << std::setw(w) << "var"
//                   << std::setw(w) << "LR_Curv" << std::setw(w) << "var"
//                   << std::setw(w) << "Width" << std::setw(w) << "var"
//                   << std::endl;
//         std::cout << "\033[0m\n";

//         // Ground Truth
//         std::vector<double> gt_values = {
//             egoPathGT->cte,
//             drivCorrGT[i].egoLaneL->cte - drivCorrGT[i].width / 2.0,
//             drivCorrGT[i].egoLaneR->cte + drivCorrGT[i].width / 2.0,
//             egoPathGT->cte,
//             egoPathGT->yaw_error,
//             drivCorrGT[i].egoLaneL->yaw_error,
//             drivCorrGT[i].egoLaneR->yaw_error,
//             egoPathGT->yaw_error,
//             egoPathGT->curvature,
//             drivCorrGT[i].egoLaneL->curvature,
//             drivCorrGT[i].egoLaneR->curvature,
//             egoPathGT->curvature,
//             drivCorrGT[i].width};

//         std::cout << std::setw(w) << "GT:";
//         for (double v : gt_values)
//             std::cout << std::setw(w) << v << std::setw(w) << "";
//         std::cout << "\n";

//         // Measurement
//         std::cout << std::setw(w) << "Meas:";
//         for (size_t j = 0; j < STATE_DIM; ++j)
//             std::cout << std::setw(w) << measurement[j].mean
//                       << std::setw(w) << measurement[j].variance;
//         std::cout << "\n";

//         // Filter Estimate
//         std::cout << std::setw(w) << "Filter:";
//         for (size_t j = 0; j < STATE_DIM; ++j)
//             std::cout << std::setw(w) << state[j].mean
//                       << std::setw(w) << state[j].variance;
//         std::cout << "\n";

//         // Error
//         std::cout << "\033[31m" << std::setw(w) << "Error:";
//         for (size_t j = 0; j < STATE_DIM; ++j)
//             std::cout << std::setw(w) << (gt_values[j] - state[j].mean)
//                       << std::setw(w) << "";
//         std::cout << "\033[0m\n";

//         std::cout << "-----------------------------------------------------------------------------------------------------------\n";

//         bayesFilter.predict(process);
//     }
// }