cmake_minimum_required(VERSION 3.14)
project(autoware_pov_scene_seg)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV 4 REQUIRED)

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
  ${TORCH_INCLUDE_DIRS}
)

link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

if(onnxruntime_USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif()


# --- Core Inference Engine ---
add_library(${PROJECT_NAME}_engine SHARED src/scene_seg.cpp)
target_include_directories(${PROJECT_NAME}_engine
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(${PROJECT_NAME}_engine
  PUBLIC
    onnxruntime
    ${OpenCV_LIBS}
)
ament_target_dependencies(${PROJECT_NAME}_engine PUBLIC)

# --- ROS2 Node ---
add_library(${PROJECT_NAME}_node SHARED src/scene_seg_node.cpp)
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
  PLUGIN "autoware_pov::AutoSeg::SceneSeg::SceneSegNode"
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