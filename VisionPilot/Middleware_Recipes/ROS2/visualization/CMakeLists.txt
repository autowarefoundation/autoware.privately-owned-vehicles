cmake_minimum_required(VERSION 3.8)
project(visualization)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Enable CUDA if available (optional)
find_package(CUDA QUIET)
if(CUDA_FOUND)
  enable_language(CUDA)
  add_definitions(-DCUDA_FOUND)
  message(STATUS "CUDA found. Enabling GPU acceleration for visualization.")
else()
  message(STATUS "CUDA not found. Using CPU-only visualization.")
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(OpenCV REQUIRED)

# visualize_masks_node
set(MASKS_SOURCES
  src/visualize_masks_node.cpp
  ../../common/visualizers/masks_visualization_engine.cpp
)

if(CUDA_FOUND)
  # Add CUDA kernel if CUDA is available
  list(APPEND MASKS_SOURCES ../../common/visualizers/cuda_visualization_kernels.cu)
  set_property(SOURCE ../../common/visualizers/cuda_visualization_kernels.cu PROPERTY LANGUAGE CUDA)
endif()

add_library(visualize_masks_node_lib SHARED ${MASKS_SOURCES})
target_include_directories(visualize_masks_node_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../common/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(visualize_masks_node_lib
  rclcpp
  rclcpp_components
  image_transport
  cv_bridge
  message_filters
  OpenCV
)

if(CUDA_FOUND)
  # Link CUDA libraries if available
  target_link_libraries(visualize_masks_node_lib ${CUDA_LIBRARIES})
  target_include_directories(visualize_masks_node_lib PRIVATE ${CUDA_INCLUDE_DIRS})
  set_target_properties(visualize_masks_node_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
rclcpp_components_register_node(
  visualize_masks_node_lib
  PLUGIN "autoware_pov::visualization::VisualizeMasksNode"
  EXECUTABLE visualize_masks_node_exe
)

# visualize_depth_node
add_library(visualize_depth_node_lib SHARED
  src/visualize_depth_node.cpp
  ../../common/visualizers/depth_visualization_engine.cpp
)
target_include_directories(visualize_depth_node_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../common/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(visualize_depth_node_lib
  rclcpp
  rclcpp_components
  image_transport
  cv_bridge
  OpenCV
)
rclcpp_components_register_node(
  visualize_depth_node_lib
  PLUGIN "autoware_pov::visualization::VisualizeDepthNode"
  EXECUTABLE visualize_depth_node_exe
)


install(
  TARGETS visualize_masks_node_lib visualize_depth_node_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()