cmake_minimum_required(VERSION 3.8)
project(pure_pursuit)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../Control/Steering/PurePursuit
    ${CMAKE_CURRENT_BINARY_DIR}/pure_pursuit_build
)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(carla_msgs REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

add_executable(pure_pursuit_node 
    src/pure_pursuit_node.cpp
    )

target_link_libraries(pure_pursuit_node pure_pursuit_lib)

target_include_directories(pure_pursuit_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(pure_pursuit_node PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

ament_target_dependencies(pure_pursuit_node rclcpp std_msgs carla_msgs)

install(TARGETS pure_pursuit_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()