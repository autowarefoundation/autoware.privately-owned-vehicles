cmake_minimum_required(VERSION 3.14)
project(models)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(CUDA REQUIRED)

# Optional: vision_msgs for AutoSpeed detection node
find_package(vision_msgs QUIET)
if(vision_msgs_FOUND)
  message(STATUS "vision_msgs found - AutoSpeed node will be built")
else()
  message(WARNING "vision_msgs not found - AutoSpeed node will be skipped. Install with: sudo apt install ros-$ENV{ROS_DISTRO}-vision-msgs")
endif()

# --- HIP Configuration (Optional) ---
# Check if HIP compiler actually exists before trying to use it
if(EXISTS "/opt/rocm/bin/hipcc" OR EXISTS "/opt/rocm-6.2.4/bin/hipcc")
  find_package(HIP QUIET)
  if(HIP_FOUND)
    message(STATUS "HIP found: ${HIP_VERSION}")
    enable_language(HIP)
    message(STATUS "HIP language enabled")
  else()
    message(STATUS "HIP not found, HIP kernels will be disabled")
  endif()
else()
  message(STATUS "HIP compiler not found, skipping HIP support")
  set(HIP_FOUND FALSE)
endif()

# --- TensorRT Configuration ---
find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS /usr/include/x86_64-linux-gnu/ /opt/tensorrt/include)
find_library(TENSORRT_LIBRARY nvinfer HINTS /usr/lib/x86_64-linux-gnu/)
find_library(TENSORRT_ONNXPARSER_LIBRARY nvonnxparser HINTS /usr/lib/x86_64-linux-gnu/)
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "Failed to find TensorRT.")
endif()

# --- ONNX Runtime Configuration ---
option(ONNXRUNTIME_ROOTDIR "Path to ONNX Runtime root directory" "")
if(NOT ONNXRUNTIME_ROOTDIR)
    message(FATAL_ERROR "You must provide the ONNX Runtime root directory via -DONNXRUNTIME_ROOTDIR")
endif()

include_directories(
  "${ONNXRUNTIME_ROOTDIR}/include"
)
link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

# --- Inference Backend Library ---
add_library(inference_backends SHARED
  ../../common/backends/onnx_runtime_backend.cpp
  ../../common/backends/tensorrt_backend.cpp
  ../../common/benchmark/fps_timer.cpp  
  ../../common/backends/autospeed/tensorrt_engine.cpp
  ../../common/backends/autospeed/tensorrt_engine.hpp
)
target_include_directories(inference_backends PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../common/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
  "${ONNXRUNTIME_ROOTDIR}/include"
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
)
target_link_libraries(inference_backends
  PUBLIC
    onnxruntime
    ${OpenCV_LIBS}
    rclcpp::rclcpp
    ${TENSORRT_LIBRARY}
    ${TENSORRT_ONNXPARSER_LIBRARY}
    ${CUDA_LIBRARIES}
)

# --- ROS2 Model Node (SceneSeg/Scene3D/DomainSeg) ---
add_library(run_model_node SHARED 
  src/run_model_node.cpp
)

# Add HIP source files if HIP is available
if(HIP_FOUND)
  target_sources(run_model_node PRIVATE
    ../../common/visualizers/masks_viz.hip.cpp
  )
  target_include_directories(run_model_node PUBLIC
    ${HIP_INCLUDE_DIRS}
  )
  target_link_libraries(run_model_node
    ${HIP_LIBRARIES}
  )
endif()

target_include_directories(run_model_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../common/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(run_model_node
  rclcpp rclcpp_components cv_bridge sensor_msgs image_transport
)
target_link_libraries(run_model_node
  inference_backends
)

rclcpp_components_register_node(
  run_model_node
  PLUGIN "autoware_pov::vision::RunModelNode"
  EXECUTABLE models_node_exe
)

# --- ROS2 AutoSpeed Node (Object Detection) ---
# Only build if vision_msgs is available
if(vision_msgs_FOUND)
  add_library(run_autospeed_node SHARED 
    src/run_autospeed_node.cpp
  )

  target_include_directories(run_autospeed_node PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../common/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../common/backends/autospeed>
    $<INSTALL_INTERFACE:include>
  )
  ament_target_dependencies(run_autospeed_node
    rclcpp rclcpp_components cv_bridge sensor_msgs vision_msgs image_transport
  )
  target_link_libraries(run_autospeed_node
    inference_backends
  )

  rclcpp_components_register_node(
    run_autospeed_node
    PLUGIN "autoware_pov::vision::RunAutoSpeedNode"
    EXECUTABLE autospeed_node_exe
  )
  
  message(STATUS "AutoSpeed node configured successfully")
endif()

# Install targets
set(INSTALL_TARGETS inference_backends run_model_node)
if(vision_msgs_FOUND)
  list(APPEND INSTALL_TARGETS run_autospeed_node)
endif()

install(
  TARGETS ${INSTALL_TARGETS}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()