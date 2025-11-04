#include "longitudinal_controller_node.hpp"

LongitudinalControllerNode::LongitudinalControllerNode(const rclcpp::NodeOptions &options)
    : Node("steering_controller_node", "", options),
      pi_controller_(1.0, 0.07)
{
  odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>("/hero/odom", 10, std::bind(&LongitudinalControllerNode::odomCallback, this, std::placeholders::_1));
  control_pub_ = this->create_publisher<std_msgs::msg::Float32>("/vehicle/throttle_cmd", 1);
  pathfinder_sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>("/pathfinder/tracked_states", 2, std::bind(&LongitudinalControllerNode::stateCallback, this, std::placeholders::_1));
  RCLCPP_INFO(this->get_logger(), "LongitudinalController Node started");
  forward_velocity_ = 0.0;
  TARGET_VEL = 22;   // 80 km/h in m/s
  ACC_LAT_MAX = 2.25; // 7.0 m/s^2
  TARGET_VEL_CAPPED = TARGET_VEL;
}

void LongitudinalControllerNode::stateCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
  if (msg->data.size() < 13)
  {
    RCLCPP_WARN(this->get_logger(), "Received message with insufficient data size: %zu", msg->data.size());
    return;
  }
  double curvature_ = msg->data[11];
  TARGET_VEL_CAPPED = std::min(TARGET_VEL, std::sqrt(ACC_LAT_MAX / std::max(std::abs(curvature_), 1e-6)));
}

void LongitudinalControllerNode::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  forward_velocity_ = msg->twist.twist.linear.x; // [m/s]
  double u = pi_controller_.computeEffort(forward_velocity_, TARGET_VEL_CAPPED);
  auto control_msg = std_msgs::msg::Float32();
  control_msg.data = std::clamp(u, -1.0, 1.0);
  control_pub_->publish(control_msg);
}

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LongitudinalControllerNode>(rclcpp::NodeOptions()));
  rclcpp::shutdown();
  return 0;
}

// [WIP] Feedforward
// throttle -> steady state speed
// 0.0      -> 0.0 m/s
// 0.1      -> 1.0 m/s
// 0.2      -> 3.6 m/s
// 0.3      -> 5.0 m/s
// 0.4      -> 8.7 m/s
// 0.5      -> 13.8 m/s
// 0.6      -> 19.7 m/s
// 0.7      -> 26.4 m/s
// 0.75     -> 29.5 m/s
// 0.8      -> 33 m/s
// 0.9      -> xx m/sS
// 1.0      -> xx m/s

// def throttle_from_exp(v, a, b):
//     if v < 0:
//         raise ValueError("speed must be non-negative")
//     val = v / a + 1.0
//     if val <= 0:
//         raise ValueError("invalid value for log; check parameters")
//     x = math.log(val) / b
//     # clamp to [0,1]
//     return max(0.0, min(1.0, x))

// # Example with previously fitted parameters:
// a = 5.94694605   # example fit result
// b = 2.37747535

