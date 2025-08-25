#include "pathfinder_node.hpp"

PathFinderNode::PathFinderNode(const rclcpp::NodeOptions &options) : Node("pathfinder_node","/pathfinder", options)
{
  this->set_parameter(rclcpp::Parameter("use_sim_time", true));
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
  init_state[12].mean = 3.5; // width assumed to be 3.5m
  bayesFilter.initialize(init_state);

  publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("tracked_states", 10);

  sub_laneL_ = this->create_subscription<nav_msgs::msg::Path>("/egoLaneL", 5,
                                                              std::bind(&PathFinderNode::callbackLaneL, this, std::placeholders::_1));
  sub_laneR_ = this->create_subscription<nav_msgs::msg::Path>("/egoLaneR", 5,
                                                              std::bind(&PathFinderNode::callbackLaneR, this, std::placeholders::_1));
  sub_path_ = this->create_subscription<nav_msgs::msg::Path>("/egoPath", 5,
                                                             std::bind(&PathFinderNode::callbackPath, this, std::placeholders::_1));

  pub_laneL_ = this->create_publisher<nav_msgs::msg::Path>("egoLaneL", 5);
  pub_laneR_ = this->create_publisher<nav_msgs::msg::Path>("egoLaneR", 5);
  pub_path_ = this->create_publisher<nav_msgs::msg::Path>("egoPath", 5);

  timer_ = this->create_wall_timer(std::chrono::milliseconds(100), std::bind(&PathFinderNode::timer_callback, this));

  RCLCPP_INFO(this->get_logger(), "PathFinder Node started");
}

void PathFinderNode::callbackLaneL(const nav_msgs::msg::Path::SharedPtr msg)
{
  left_coeff = pathMsg2Coeff(msg);
}

void PathFinderNode::callbackLaneR(const nav_msgs::msg::Path::SharedPtr msg)
{
  right_coeff = pathMsg2Coeff(msg);
}

void PathFinderNode::callbackPath(const nav_msgs::msg::Path::SharedPtr msg)
{
  path_coeff = pathMsg2Coeff(msg);
}

std::array<double, 3UL> PathFinderNode::pathMsg2Coeff(const nav_msgs::msg::Path::SharedPtr &msg)
{
  auto now = this->get_clock()->now();
  auto msg_time = rclcpp::Time(msg->header.stamp);
  rclcpp::Duration threshold(0, 60000000); // (sec, nanosec)
  if ((now - msg_time) > threshold)
  {
    RCLCPP_WARN(this->get_logger(), "Dropping stale message %ld nsec old", (now - msg_time).nanoseconds());
    return {0.0, 0.0, 0.0};
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
  auto drivCorr = drivingCorridor(fittedCurve(left_coeff), fittedCurve(right_coeff), fittedCurve(path_coeff));

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
  auto path = nav_msgs::msg::Path();
  auto laneL = nav_msgs::msg::Path();
  auto laneR = nav_msgs::msg::Path();
  auto header = std_msgs::msg::Header();
  header.stamp = this->get_clock()->now();
  header.frame_id = "hero";
  path.header = header;
  laneL.header = header;
  laneR.header = header;

  double res = 0.1; // resolution in meters
  for (double i=0;i < 30;i+=res)
  {
    auto p1 = geometry_msgs::msg::PoseStamped();
    p1.header = header;
    p1.pose.position.x = i;
    p1.pose.position.y = i * i * path_coeff[0] + i * path_coeff[1] + path_coeff[2];
    path.poses.push_back(p1);
    
    auto p2 = geometry_msgs::msg::PoseStamped();
    p2.header = header;
    p2.pose.position.x = i;
    p2.pose.position.y = i * i * left_coeff[0] + i * left_coeff[1] + left_coeff[2];
    laneL.poses.push_back(p2);

    auto p3 = geometry_msgs::msg::PoseStamped();
    p3.header = header;
    p3.pose.position.x = i;
    p3.pose.position.y = i * i * right_coeff[0] + i * right_coeff[1] + right_coeff[2];
    laneR.poses.push_back(p3);
  }
  pub_path_->publish(path);
  pub_laneL_->publish(laneL);
  pub_laneR_->publish(laneR);

}

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PathFinderNode>(rclcpp::NodeOptions()));
  rclcpp::shutdown();
  return 0;
}
