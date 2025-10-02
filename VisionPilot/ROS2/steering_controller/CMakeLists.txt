cmake_minimum_required(VERSION 3.8)
project(steering_controller)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../Control/Steering/SteeringController
    ${CMAKE_CURRENT_BINARY_DIR}/steering_controller_build
)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(carla_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

add_executable(steering_controller_node 
    src/steering_controller_node.cpp
    )

target_link_libraries(steering_controller_node steering_controller_lib)

target_include_directories(steering_controller_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(steering_controller_node PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

ament_target_dependencies(steering_controller_node rclcpp std_msgs carla_msgs nav_msgs)

install(TARGETS steering_controller_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()