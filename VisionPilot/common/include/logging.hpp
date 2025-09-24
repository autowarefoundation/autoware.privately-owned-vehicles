#ifndef LOGGING_HPP_
#define LOGGING_HPP_

#define LOG_TYPE_NONE  0
#define LOG_TYPE_ROS   1

#ifndef LOG_TYPE
  #define LOG_TYPE LOG_TYPE_ROS
#endif

#if LOG_TYPE == LOG_TYPE_ROS
  #include "rclcpp/rclcpp.hpp"
  #define LOG_INFO(...) \
    RCLCPP_INFO(rclcpp::get_logger("common"), __VA_ARGS__)  
  #define LOG_WARN(...) \
    RCLCPP_WARN(rclcpp::get_logger("common"), __VA_ARGS__)  
  #define LOG_ERROR(...) \
    RCLCPP_ERROR(rclcpp::get_logger("common"), __VA_ARGS__)
#else
  #include <cstdio>
  #define LOG_INFO(...) printf(__VA_ARGS__)
  #define LOG_WARN(...) printf(__VA_ARGS__)
  #define LOG_ERROR(...) printf(__VA_ARGS__)
#endif

#endif // LOGGING_HPP_
