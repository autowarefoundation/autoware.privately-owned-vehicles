#include "pathfinder_node.hpp"

// TODO: track and update curvature over time instead of only at current timestep

PathFinderNode::PathFinderNode(const rclcpp::NodeOptions &options) : Node("pathfinder_node", "/pathfinder", options)
{
  bayesFilter = Estimator();
  bayesFilter.configureFusionGroups({
      // {start_idx,end_idx}
      // fuse indices 1,2 → result in index 3
      {0, 3}, // cte1, cte2 → fused at index 3
      {5, 7}, // yaw1, yaw2 → fused at index 7
      {9, 11} // curv1, curv2 → fused at index 11
  });
  Gaussian default_state = {0.0, 1e3}; // states can be any value, variance is large
  std::array<Gaussian, STATE_DIM> init_state;
  init_state.fill(default_state);
  init_state[12].mean = LANE_WIDTH;    // assumed lane width
  init_state[12].variance = 0.5 * 0.5; // lane width variance
  bayesFilter.initialize(init_state);

  publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("tracked_states", 3);
  corr_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("driving_corridor", 3);

  sub_laneL_ = this->create_subscription<nav_msgs::msg::Path>("/egoLaneL", 3,
                                                              std::bind(&PathFinderNode::callbackLaneL, this, std::placeholders::_1));
  sub_laneR_ = this->create_subscription<nav_msgs::msg::Path>("/egoLaneR", 3,
                                                              std::bind(&PathFinderNode::callbackLaneR, this, std::placeholders::_1));
  sub_path_ = this->create_subscription<nav_msgs::msg::Path>("/egoPath", 3,
                                                             std::bind(&PathFinderNode::callbackPath, this, std::placeholders::_1));

  pub_laneL_ = this->create_publisher<nav_msgs::msg::Path>("egoLaneL", 3);
  pub_laneR_ = this->create_publisher<nav_msgs::msg::Path>("egoLaneR", 3);
  pub_path_ = this->create_publisher<nav_msgs::msg::Path>("egoPath", 3);

  timer_ = rclcpp::create_timer(this->get_node_base_interface(),
                                this->get_node_timers_interface(),
                                this->get_clock(),
                                std::chrono::milliseconds(10), 
                                std::bind(&PathFinderNode::timer_callback, this));

  left_msg = std::make_shared<nav_msgs::msg::Path>();
  right_msg = std::make_shared<nav_msgs::msg::Path>();
  path_msg = std::make_shared<nav_msgs::msg::Path>();
  left_msg->header.stamp = this->get_clock()->now();
  right_msg->header.stamp = this->get_clock()->now();
  path_msg->header.stamp = this->get_clock()->now();

  RCLCPP_INFO(this->get_logger(), "PathFinder Node started");
}

void PathFinderNode::callbackLaneL(const nav_msgs::msg::Path::SharedPtr msg)
{
  left_msg = msg;
}

void PathFinderNode::callbackLaneR(const nav_msgs::msg::Path::SharedPtr msg)
{
  right_msg = msg;
}

void PathFinderNode::callbackPath(const nav_msgs::msg::Path::SharedPtr msg)
{
  path_msg = msg;
}

std::array<double, 3> PathFinderNode::pathMsg2Coeff(const nav_msgs::msg::Path::SharedPtr &msg)
{
  auto now = this->get_clock()->now();
  auto msg_time = rclcpp::Time(msg->header.stamp);
  rclcpp::Duration threshold(0, 100000000); // (sec, nanosec)
  if ((now - msg_time) > threshold)
  {
    RCLCPP_WARN(this->get_logger(), "Dropping stale message %ld nsec old", (now - msg_time).nanoseconds());
    return {std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN()};
  }
  std::vector<cv::Point2f> points;
  for (const auto &pose : msg->poses)
  {
    points.emplace_back(pose.pose.position.y, pose.pose.position.x);
  }
  return fitQuadPoly(points);
}

void PathFinderNode::timer_callback()
{
  auto left_coeff = pathMsg2Coeff(left_msg);
  auto right_coeff = pathMsg2Coeff(right_msg);
  auto path_coeff = pathMsg2Coeff(path_msg);
  std::array<Gaussian, STATE_DIM> measurement;
  std::array<Gaussian, STATE_DIM> process;
  std::random_device rd;
  std::default_random_engine generator(rd());
  std::uniform_real_distribution<double> dist(-epsilon, epsilon);
  for (size_t i = 0; i < STATE_DIM; ++i)
  {
    process[i].mean = dist(generator);
    // process[i].variance = proc_SD * proc_SD;
    // measurement[i].variance = meas_SD * meas_SD;
  }
  double STD_P_YAW = 0.2;    // rad
  double STD_P_CURV = 0.05;  // 1/m
  double STD_P_CTE = 0.5;    // m
  double STD_P_WIDTH = 0.01; // m

  double STD_M_YAW = 0.01;            // rad
  double STD_M_CURV = 2 * STD_P_CURV; // 1/m
  double STD_M_CTE = 0.1;             // m
  double STD_M_WIDTH = 0.01;          // m

  process[0].variance = STD_P_CTE * STD_P_CTE;
  process[1].variance = STD_P_CTE * STD_P_CTE;
  process[2].variance = STD_P_CTE * STD_P_CTE;

  process[4].variance = STD_P_YAW * STD_P_YAW;
  process[5].variance = STD_P_YAW * STD_P_YAW;
  process[6].variance = STD_P_YAW * STD_P_YAW;

  process[8].variance = STD_P_CURV * STD_P_CURV;
  process[9].variance = STD_P_CURV * STD_P_CURV;
  process[10].variance = STD_P_CURV * STD_P_CURV;

  process[12].variance = STD_P_WIDTH * STD_P_WIDTH;

  measurement[0].variance = STD_M_CTE * STD_M_CTE;
  measurement[1].variance = STD_M_CTE * STD_M_CTE;
  measurement[2].variance = STD_M_CTE * STD_M_CTE;

  measurement[4].variance = STD_M_YAW * STD_M_YAW;
  measurement[5].variance = STD_M_YAW * STD_M_YAW;
  measurement[6].variance = STD_M_YAW * STD_M_YAW;

  measurement[8].variance = STD_M_CURV * STD_M_CURV;
  measurement[9].variance = STD_M_CURV * STD_M_CURV;
  measurement[10].variance = STD_M_CURV * STD_M_CURV;

  measurement[12].variance = STD_M_WIDTH * STD_M_WIDTH;

  auto width = bayesFilter.getState()[12].mean;
  fittedCurve egoPath(path_coeff);
  fittedCurve egoLaneL(left_coeff);
  fittedCurve egoLaneR(right_coeff);

  // cte to be fused wrt lane center
  measurement[0].mean = egoPath.cte;
  measurement[1].mean = egoLaneL.cte + width / 2.0;
  measurement[2].mean = egoLaneR.cte - width / 2.0;
  measurement[3].mean = std::numeric_limits<double>::quiet_NaN();

  measurement[4].mean = egoPath.yaw_error;
  measurement[5].mean = egoLaneL.yaw_error;
  measurement[6].mean = egoLaneR.yaw_error;
  measurement[7].mean = std::numeric_limits<double>::quiet_NaN();

  measurement[8].mean = egoPath.curvature;
  measurement[9].mean = egoLaneL.curvature;
  measurement[10].mean = egoLaneR.curvature;
  measurement[11].mean = std::numeric_limits<double>::quiet_NaN();

  for (int i = 0; i < STATE_DIM; i++)
  {
    if (std::isnan(measurement[i].mean))
    {
      RCLCPP_WARN(this->get_logger(), "Measurement %d is NaN", i);
    }
  }

  if (std::isnan(egoLaneL.cte) && std::isnan(egoLaneR.cte))
  {
    // both lane missing, keep width
    measurement[12].mean = LANE_WIDTH;
  }
  else if (std::isnan(egoLaneL.cte))
  {
    // left lane missing, use right lane to infer left lane
    measurement[12].mean = 2 * (egoLaneR.cte - egoPath.cte);
  }
  else if (std::isnan(egoLaneR.cte))
  {
    // right lane missing, use left lane to infer right lane
    measurement[12].mean = -2 * (egoLaneL.cte - egoPath.cte);
  }
  else
  {
    // both lanes present, use lane width directly
    measurement[12].mean = egoLaneR.cte - egoLaneL.cte;
  }

  bayesFilter.update(measurement);
  const auto &state = bayesFilter.getState();
  bayesFilter.predict(process);

  std::string mean_log_msg = "Mean: [";
  std::string var_log_msg = "Var:  [";
  for (int i = 0; i < STATE_DIM; i++)
  {
    if (i != 0 && i % 4 == 0 || i % 4 == 3)
    {
      mean_log_msg += "| ";
      var_log_msg += "| ";
    }
    mean_log_msg += std::to_string(state[i].mean) + " ";
    var_log_msg += std::to_string(state[i].variance) + " ";
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

  auto path = nav_msgs::msg::Path();
  auto laneL = nav_msgs::msg::Path();
  auto laneR = nav_msgs::msg::Path();
  auto header = std_msgs::msg::Header();
  header.stamp = this->get_clock()->now();
  header.frame_id = "hero";
  path.header = header;
  laneL.header = header;
  laneR.header = header;

  // double res = 0.1; // resolution in meter
  // for (double i = 0; i < 30; i += res)
  // {
  //   auto p1 = geometry_msgs::msg::PoseStamped();
  //   p1.header = header;
  //   p1.pose.position.x = i;
  //   p1.pose.position.y = i * i * path_coeff[0] + i * path_coeff[1] + path_coeff[2];
  //   path.poses.push_back(p1);

  //   auto p2 = geometry_msgs::msg::PoseStamped();
  //   p2.header = header;
  //   p2.pose.position.x = i;
  //   p2.pose.position.y = i * i * left_coeff[0] + i * left_coeff[1] + left_coeff[2];
  //   laneL.poses.push_back(p2);

  //   auto p3 = geometry_msgs::msg::PoseStamped();
  //   p3.header = header;
  //   p3.pose.position.x = i;
  //   p3.pose.position.y = i * i * right_coeff[0] + i * right_coeff[1] + right_coeff[2];
  //   laneR.poses.push_back(p3);
  // }
  // pub_path_->publish(path);
  // pub_laneL_->publish(laneL);
  // pub_laneR_->publish(laneR);

  // publishLaneMarker(state[12].mean, state[0].mean, state[4].mean, state[8].mean, {0.0, 1.0, 0.0, 0.8});  // raw egoPath
  // publishLaneMarker(state[12].mean, state[1].mean, state[5].mean, state[9].mean, {1.0, 0.0, 0.0, 0.8});  // raw egoLaneL
  // publishLaneMarker(state[12].mean, state[2].mean, state[6].mean, state[10].mean, {0.0, 0.0, 1.0, 0.8}); // raw egoLaneR
  publishLaneMarker(state[12].mean, state[3].mean, state[7].mean, state[11].mean, {1.0, 1.0, 1.0, 0.8}); // fused
}

void PathFinderNode::publishLaneMarker(double lane_width, double cte, double yaw_error, double curvature, std::array<float, 4> rgba)
{
  auto lane_marker = visualization_msgs::msg::Marker();
  lane_marker.header.stamp = this->get_clock()->now();
  lane_marker.header.frame_id = "hero";
  lane_marker.ns = "tracked_lane";
  lane_marker.id = 0;
  lane_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
  lane_marker.action = visualization_msgs::msg::Marker::ADD;

  lane_marker.scale.x = lane_width;
  lane_marker.color.r = rgba[0];
  lane_marker.color.g = rgba[1];
  lane_marker.color.b = rgba[2];
  lane_marker.color.a = rgba[3];

  // Generate a circular arc based on curvature
  const double radius = (std::abs(curvature) > 1e-9) ? 1.0 / curvature : 1e9;
  const double arc_length = 20.0; // visualize 20 m ahead
  const int num_points = 50;

  for (int i = 0; i < num_points; ++i)
  {
    double s = (arc_length / (num_points - 1)) * i;
    geometry_msgs::msg::Point p;
    p.x = s * std::cos(-yaw_error);
    p.y = -cte + radius * (1 - std::cos(s / radius)); // simple curvature offset
    p.z = 0.0;
    lane_marker.points.push_back(p);
  }
  corr_pub_->publish(lane_marker);
}

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PathFinderNode>(rclcpp::NodeOptions()));
  rclcpp::shutdown();
  return 0;
}
