#include "pure_pursuit_node.hpp"

PurePursuitNode::PurePursuitNode(const rclcpp::NodeOptions &options) : Node("pure_pursuit_node", "", options), pp(2.85)
{

  this->set_parameter(rclcpp::Parameter("use_sim_time", true));
  sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>("/pathfinder/tracked_states", 10, std::bind(&PurePursuitNode::computeSteering, this, std::placeholders::_1));
  test_sub_ = this->create_subscription<nav_msgs::msg::Path>("/egoPath", 10, std::bind(&PurePursuitNode::testComputeSteering, this, std::placeholders::_1));
  steering_pub_ = this->create_publisher<carla_msgs::msg::CarlaEgoVehicleControl>("carla/hero/vehicle_control_cmd", 10);
  RCLCPP_INFO(this->get_logger(), "PurePursuit Node started");
}

void PurePursuitNode::testComputeSteering(const nav_msgs::msg::Path msg)
{
  double wheelbase = 2.85;
  double forward_velocity = 10.0; // Placeholder for actual velocity input
  double lookahead_distance_ = 2.5;// + forward_velocity * 0.5;
  double yaw_error = 0.0;
  for (const auto &pose : msg.poses) {
    double x = pose.pose.position.x;
    double y = pose.pose.position.y;
    double dist = std::sqrt(x * x + y * y);
    if (dist >= lookahead_distance_){
      yaw_error = std::atan2(y, x);
      break;
    }
  }

  double steering_angle = std::atan2(2 * wheelbase * std::sin(yaw_error), lookahead_distance_);//+ std::atan2(curvature * wheelbase_, 1.0);

  auto control_msg = carla_msgs::msg::CarlaEgoVehicleControl();
  control_msg.header.stamp = this->now();
  control_msg.steer = - steering_angle /(70.0 * (M_PI / 180.0)) ;
  control_msg.throttle = 0.3; // relate to velocity
  control_msg.brake = 0.0;
  control_msg.hand_brake = false;
  control_msg.reverse = false;
  control_msg.manual_gear_shift = false;
  steering_pub_->publish(control_msg);

  RCLCPP_INFO(this->get_logger(), "Computed steering angle: %f", steering_angle);
}

void PurePursuitNode::computeSteering(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
  if (msg->data.size() < 13)
  {
    RCLCPP_WARN(this->get_logger(), "Received message with insufficient data size: %zu", msg->data.size());
    return;
  }
  double cte = msg->data[3];
  double yaw_error = msg->data[7];
  double curvature = msg->data[11];
  double forward_velocity = 10.0; // Placeholder for actual velocity input
  double steering_angle = pp.computeSteering(cte, yaw_error, curvature, forward_velocity);

  auto control_msg = carla_msgs::msg::CarlaEgoVehicleControl();
  control_msg.header.stamp = this->now();
  control_msg.steer = - steering_angle /(70.0 * (M_PI / 180.0)) ;
  control_msg.throttle = 0.3; // relate to velocity
  control_msg.brake = 0.0;
  control_msg.hand_brake = false;
  control_msg.reverse = false;
  control_msg.manual_gear_shift = false;
  // steering_pub_->publish(control_msg);

  // RCLCPP_INFO(this->get_logger(), "Computed steering angle: %f", steering_angle);
}

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PurePursuitNode>(rclcpp::NodeOptions()));
  rclcpp::shutdown();
  return 0;
}