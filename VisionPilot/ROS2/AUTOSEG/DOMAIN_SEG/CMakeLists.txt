cmake_minimum_required(VERSION 3.14)
project(autoware_pov_domain_seg)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(CUDA REQUIRED)

# --- TensorRT Configuration ---
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS
    /usr/include/x86_64-linux-gnu/
    /usr/local/cuda/include
    /opt/tensorrt/include
)
find_library(TENSORRT_LIBRARY nvinfer HINTS /usr/lib/x86_64-linux-gnu/ /usr/local/cuda/lib64)
find_library(TENSORRT_ONNXPARSER_LIBRARY nvonnxparser HINTS /usr/lib/x86_64-linux-gnu/ /usr/local/cuda/lib64)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "Failed to find TensorRT.")
endif()

message(STATUS "Found TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "Found TensorRT library: ${TENSORRT_LIBRARY}")

# --- ONNX Runtime Configuration ---
option(ONNXRUNTIME_ROOTDIR "Path to ONNX Runtime root directory" "")
if(NOT ONNXRUNTIME_ROOTDIR)
    message(FATAL_ERROR "You must provide the ONNX Runtime root directory via -DONNXRUNTIME_ROOTDIR")
endif()

include_directories(
  "${ONNXRUNTIME_ROOTDIR}/include"
  "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime"
  "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session"
  "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/providers/dnnl"
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
)

link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

if(onnxruntime_USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif()

# --- Core Inference Engine ---
add_library(${PROJECT_NAME}_engine SHARED
  src/domain_seg.cpp
  src/onnx_runtime_backend.cpp
  src/tensorrt_backend.cpp
)
target_include_directories(${PROJECT_NAME}_engine
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(${PROJECT_NAME}_engine
  PUBLIC
    onnxruntime
    ${OpenCV_LIBS}
    rclcpp::rclcpp
    ${TENSORRT_LIBRARY}
    ${TENSORRT_ONNXPARSER_LIBRARY}
    ${CUDA_LIBRARIES}
)

# --- ROS2 Node ---
add_library(${PROJECT_NAME}_node SHARED src/domain_seg_node.cpp)
target_link_libraries(${PROJECT_NAME}_node
  PRIVATE
    ${PROJECT_NAME}_engine
)
ament_target_dependencies(${PROJECT_NAME}_node
  PUBLIC
    rclcpp
    rclcpp_components
    cv_bridge
    image_transport
    sensor_msgs
)

# --- Register Component ---
rclcpp_components_register_node(
  ${PROJECT_NAME}_node
  PLUGIN "autoware_pov::AutoSeg::DomainSeg::DomainSegNode"
  EXECUTABLE ${PROJECT_NAME}_node_exe
)

# --- Installation ---
install(
  TARGETS
    ${PROJECT_NAME}_engine
    ${PROJECT_NAME}_node
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
  PROGRAMS scripts/video_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package() 