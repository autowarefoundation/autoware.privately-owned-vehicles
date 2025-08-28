cmake_minimum_required(VERSION 3.8)
project(PATHFINDER)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)

# Include OpenCV and your own headers
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include/PATHFINDER
    ${EIGEN3_INCLUDE_DIR}
)

add_executable(pathfinder_node 
    src/pathfinder_node.cpp
    src/path_finder.cpp
    src/estimator.cpp
    )

target_link_libraries(pathfinder_node ${OpenCV_LIBS} yaml-cpp)

target_include_directories(pathfinder_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(pathfinder_node PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

ament_target_dependencies(pathfinder_node rclcpp std_msgs nav_msgs)

install(TARGETS pathfinder_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()