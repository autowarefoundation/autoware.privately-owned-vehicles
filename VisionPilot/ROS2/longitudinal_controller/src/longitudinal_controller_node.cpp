#include "longitudinal_controller_node.hpp"

LongitudinalControllerNode::LongitudinalControllerNode(const rclcpp::NodeOptions &options)
    : Node("steering_controller_node", "", options),
      pi_throttle_(0.05, 0.000, 0.5),
      pi_brake_(0.005, 0.0, 0.0)
{
  odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>("/hero/odom", 10, std::bind(&LongitudinalControllerNode::odomCallback, this, std::placeholders::_1));
  imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>("/carla/hero/imu", 10, std::bind(&LongitudinalControllerNode::imuCallback, this, std::placeholders::_1));
  throttle_pub_ = this->create_publisher<std_msgs::msg::Float32>("/vehicle/throttle_cmd", 1);
  brake_pub_ = this->create_publisher<std_msgs::msg::Float32>("/vehicle/brake_cmd", 1);
  pathfinder_sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>("/pathfinder/tracked_states", 2, std::bind(&LongitudinalControllerNode::stateCallback, this, std::placeholders::_1));
  RCLCPP_INFO(this->get_logger(), "LongitudinalController Node started");
  forward_velocity_ = 0.0;
  longitudinal_acceleration_ = 0.0;
  TARGET_VEL = 23.6;  // 80 km/h in m/s, actual 25m/s after adding steady state error
  ACC_LAT_MAX = 2.25; // 7.0 m/s^2
  ACC_LONG_MAX = 5.0;
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

void LongitudinalControllerNode::imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
  longitudinal_acceleration_ = msg->linear_acceleration.x; // [m/s^2]
}

void LongitudinalControllerNode::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  auto throttle_msg = std_msgs::msg::Float32();
  auto brake_msg = std_msgs::msg::Float32();
  brake_msg.data = 0.0;
  throttle_msg.data = 0.0;

  forward_velocity_ = msg->twist.twist.linear.x; // [m/s]
  double feedforward_u = (forward_velocity_ > TARGET_VEL_CAPPED) ? 0 : vel_to_throttle(TARGET_VEL_CAPPED);
  // double TARGET_ACCEL = TARGET_VEL_CAPPED > forward_velocity_ ? (TARGET_VEL_CAPPED - forward_velocity_) / 0.02 : 0.0; // determine how much to brake
  double TARGET_ACCEL = (TARGET_VEL_CAPPED - forward_velocity_) / 0.02; // determine how much to brake

  double brake = pi_brake_.computeEffort(-longitudinal_acceleration_, -TARGET_ACCEL);
  double throttle = feedforward_u;
  // double throttle = pi_throttle_.computeEffort(longitudinal_acceleration_, TARGET_ACCEL);

  if (TARGET_ACCEL < -2.5)
  {
    brake_msg.data = std::clamp(brake, 0.0, 1.0);
  }

  throttle_msg.data = std::clamp(throttle, 0.0, 1.0);
  throttle_pub_->publish(throttle_msg);
  brake_pub_->publish(brake_msg);
}

double LongitudinalControllerNode::vel_to_throttle(double v)
{
  double a = 5.94694605;
  double b = 2.37747535;
  if (v < 0)
  {
    throw std::invalid_argument("speed must be non-negative");
  }
  double val = v / a + 1.0;
  if (val <= 0)
  {
    throw std::invalid_argument("invalid value for log; check parameters");
  }
  double x = std::log(val) / b;
  // clamp to [0,1]
  return std::clamp(x, 0.0, 1.0);
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
