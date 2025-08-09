#include "pathfinder_node.hpp"

PathFinderNode::PathFinderNode() : Node("pathfinder_node"), drivCorr(std::nullopt, std::nullopt, std::nullopt)
{
  bayesFilter = Estimator();
  bayesFilter.configureFusionGroups({
      // {start_idx,end_idx}
      // fuse indices 1,2 → result in index 3
      {0, 3}, // cte1, cte2 → fused at index 3
      {5, 7}, // yaw1, yaw2 → fused at index 7
      {9, 11} // curv1, curv2 → fused at index 11
  });
  Gaussian default_state = {0.0, 1e6}; // states can be any value, variance is large
  std::array<Gaussian, STATE_DIM> init_state;
  init_state.fill(default_state);
  bayesFilter.initialize(init_state);

  publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("tracked_states", 10);
  subscription_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
      "egoLanes", 10, std::bind(&PathFinderNode::topic_callback, this, std::placeholders::_1));

  // For loopback testing without egoLanes and egoPath topics
  loopback_publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("egoLanes", 10);
  timer_ = this->create_wall_timer(std::chrono::milliseconds(1000), std::bind(&PathFinderNode::timer_callback_, this));

  std::string package_share_dir = ament_index_cpp::get_package_share_directory("PATHFINDER");
  yaml_fp = package_share_dir + "/test/000001/1616005402699.yaml";
  homo_fp = package_share_dir + "/test/image_to_world_transform.yaml";
  estimateH(homo_fp);
  H = loadHFromYaml(homo_fp);

  RCLCPP_INFO(this->get_logger(), "PathFinder Node started");
}

void PathFinderNode::topic_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
  std::vector<cv::Point2f> list1;
  std::vector<cv::Point2f> list2;

  // Check if layout has correct dimensions
  if (msg->layout.dim.size() != 3)
  {
    RCLCPP_ERROR(rclcpp::get_logger("unpack"), "Invalid layout: expected 3 dimensions");
    return;
  }

  size_t num_lists = msg->layout.dim[0].size;  // Should be 2
  size_t max_points = msg->layout.dim[1].size; // max points per list
  size_t xy_size = msg->layout.dim[2].size;    // Should be 2

  if (num_lists != 2 || xy_size != 2)
  {
    RCLCPP_ERROR(rclcpp::get_logger("unpack"), "Unexpected layout sizes");
    return;
  }

  const auto &data = msg->data;

  if (data.size() < num_lists * max_points * xy_size)
  {
    RCLCPP_ERROR(rclcpp::get_logger("unpack"), "Data size does not match layout");
    return;
  }

  // Reserve memory
  list1.reserve(max_points);
  list2.reserve(max_points);

  // Parse first list
  size_t offset_list1 = 0;
  size_t offset_list2 = max_points * xy_size; // second list starts after first list block

  for (size_t i = 0; i < max_points; ++i)
  {
    float x1 = data[offset_list1 + i * xy_size];
    float y1 = data[offset_list1 + i * xy_size + 1];

    if (!std::isnan(x1) && !std::isnan(y1))
    {
      list1.emplace_back(x1, y1);
    }

    float x2 = data[offset_list2 + i * xy_size];
    float y2 = data[offset_list2 + i * xy_size + 1];

    if (!std::isnan(x2) && !std::isnan(y2))
    {
      list2.emplace_back(x2, y2);
    }
  }
  auto coeff1 = fitQuadPoly(list1);
  auto coeff2 = fitQuadPoly(list2);

  drivCorr = drivingCorridor(fittedCurve(coeff1), fittedCurve(coeff2), std::nullopt);

  std::array<Gaussian, STATE_DIM> measurement;
  std::array<Gaussian, STATE_DIM> process;

  std::random_device rd;
  std::default_random_engine generator(rd());
  std::uniform_real_distribution<double> dist(-epsilon, epsilon);
  for (size_t i = 0; i < STATE_DIM; ++i)
  {
    process[i].mean = dist(generator);
    process[i].variance = proc_SD * proc_SD;
    measurement[i].variance = meas_SD * meas_SD;
  }

  const auto &egoPath = drivCorr.egoPath;
  const auto &egoLaneL = drivCorr.egoLaneL;
  const auto &egoLaneR = drivCorr.egoLaneR;

  measurement[0].mean = egoPath->cte;
  measurement[1].mean = egoLaneL->cte - drivCorr.width / 2.0;
  measurement[2].mean = egoLaneR->cte + drivCorr.width / 2.0;
  measurement[3].mean = 0.0;

  measurement[4].mean = egoPath->yaw_error;
  measurement[5].mean = egoLaneL->yaw_error;
  measurement[6].mean = egoLaneR->yaw_error;
  measurement[7].mean = 0.0;

  measurement[8].mean = egoPath->curvature;
  measurement[9].mean = egoLaneL->curvature;
  measurement[10].mean = egoLaneR->curvature;
  measurement[11].mean = 0.0;

  measurement[12].mean = drivCorr.width;

  bayesFilter.update(measurement);
  const auto &state = bayesFilter.getState();
  bayesFilter.predict(process);

  std::string mean_log_msg = "Mean: [";
  std::string var_log_msg = "Var:  [";
  for (auto s : state)
  {
    mean_log_msg += std::to_string(s.mean) + "  ";
    var_log_msg += std::to_string(s.variance) + "  ";
  }
  mean_log_msg += "]";
  var_log_msg += "]";
  RCLCPP_INFO(this->get_logger(), mean_log_msg.c_str());
  RCLCPP_INFO(this->get_logger(), var_log_msg.c_str());

  auto out_msg = std_msgs::msg::Float32MultiArray();
  out_msg.data.resize(STATE_DIM * 2);
  out_msg.layout.dim.push_back(std_msgs::msg::MultiArrayDimension());
  out_msg.layout.dim[0].label = "mean";
  out_msg.layout.dim[0].size = STATE_DIM;
  out_msg.layout.dim[0].stride = 1;
  out_msg.layout.data_offset = 0;

  out_msg.layout.dim.push_back(std_msgs::msg::MultiArrayDimension());
  out_msg.layout.dim[1].label = "variance";
  out_msg.layout.dim[1].size = STATE_DIM;
  out_msg.layout.dim[1].stride = 1;

  for (size_t i = 0; i < STATE_DIM; ++i)
  {
    out_msg.data[i] = state[i].mean;
    out_msg.data[STATE_DIM + i] = state[i].variance;
  }
  publisher_->publish(out_msg);
}

void PathFinderNode::timer_callback_()
{
  auto egoLanesPts = loadLanesFromYaml(yaml_fp, H);
  if (egoLanesPts.size() < 2)
  {
    std::cerr << "Not enough lanes to calculate ego path" << std::endl;
    return;
  }
  auto list1 = egoLanesPts[2].BevPoints;
  auto list2 = egoLanesPts[1].BevPoints;
  auto msg = std_msgs::msg::Float32MultiArray();

  size_t max_points = std::max(list1.size(), list2.size());
  msg.data.reserve(2 * max_points * 2); // 2 lists * max_points * (x,y)

  float NaN = std::numeric_limits<float>::quiet_NaN();

  for (auto &p : list1)
  {
    msg.data.push_back(static_cast<float>(p.x));
    msg.data.push_back(static_cast<float>(p.y));
  }
  for (size_t i = list1.size(); i < max_points; ++i)
  {
    msg.data.push_back(NaN);
    msg.data.push_back(NaN);
  }

  for (auto &p : list2)
  {
    msg.data.push_back(static_cast<float>(p.x));
    msg.data.push_back(static_cast<float>(p.y));
  }
  for (size_t i = list2.size(); i < max_points; ++i)
  {
    msg.data.push_back(NaN);
    msg.data.push_back(NaN);
  }

  // Layout
  msg.layout.dim.resize(3);

  // dim[0]: lists
  msg.layout.dim[0].label = "lists";
  msg.layout.dim[0].size = 2;                    // 2 lists
  msg.layout.dim[0].stride = max_points * 2 * 2; // each list = max_points * (x,y)

  // dim[1]: points
  msg.layout.dim[1].label = "points";
  msg.layout.dim[1].size = max_points; // max_points per list
  msg.layout.dim[1].stride = 2;        // each point has 2 elements

  // dim[2]: xy
  msg.layout.dim[2].label = "xy";
  msg.layout.dim[2].size = 2;
  msg.layout.dim[2].stride = 1; // x and y are adjacent

  msg.layout.data_offset = 0;

  loopback_publisher_->publish(msg);
}

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PathFinderNode>());
  rclcpp::shutdown();
  return 0;
}
