cmake_minimum_required(VERSION 3.8)
project(sensors)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(include ../../common/include)

# video_publisher_node
add_library(video_publisher_node SHARED
  src/video_publisher_node.cpp
)
ament_target_dependencies(video_publisher_node
  rclcpp
  rclcpp_components
  sensor_msgs
  cv_bridge
  OpenCV
)
rclcpp_components_register_node(
  video_publisher_node
  PLUGIN "autoware_pov::sensors::VideoPublisherNode"
  EXECUTABLE video_publisher_node_exe
)

install(
  TARGETS video_publisher_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include/
  DESTINATION include
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()