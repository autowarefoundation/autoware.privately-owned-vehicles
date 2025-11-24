cmake_minimum_required(VERSION 3.8)
project(longitudinal_controller)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../Control/Longitudinal/PI_Controller
    ${CMAKE_CURRENT_BINARY_DIR}/pi_controller_build
)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

add_executable(longitudinal_controller_node 
    src/longitudinal_controller_node.cpp
    )

target_link_libraries(longitudinal_controller_node pi_controller_lib)

target_include_directories(longitudinal_controller_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(longitudinal_controller_node PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

ament_target_dependencies(longitudinal_controller_node rclcpp std_msgs nav_msgs sensor_msgs)

install(TARGETS longitudinal_controller_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()