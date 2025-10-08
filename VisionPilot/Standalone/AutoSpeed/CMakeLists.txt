cmake_minimum_required(VERSION 3.16)
project(autospeed_standalone LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# GStreamer
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GSTREAMER_APP REQUIRED gstreamer-app-1.0)

# TensorRT
set(TENSORRT_ROOT "/usr/local/tensorrt" CACHE PATH "TensorRT installation directory")
if(NOT EXISTS ${TENSORRT_ROOT})
    set(TENSORRT_ROOT "/usr")
endif()

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT}/include
    PATH_SUFFIXES include)
find_library(TENSORRT_LIBRARY nvinfer
    HINTS ${TENSORRT_ROOT}/lib
    PATH_SUFFIXES lib lib64)
find_library(TENSORRT_ONNX_LIBRARY nvonnxparser
    HINTS ${TENSORRT_ROOT}/lib
    PATH_SUFFIXES lib lib64)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY OR NOT TENSORRT_ONNX_LIBRARY)
    message(FATAL_ERROR "TensorRT not found")
endif()

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${GSTREAMER_INCLUDE_DIRS}
    ${GSTREAMER_APP_INCLUDE_DIRS}
    ../../common/include
)

# AutoSpeed TensorRT backend
add_library(autospeed_backend
    ../../common/backends/autospeed/tensorrt_engine.cpp
)
target_compile_definitions(autospeed_backend PRIVATE LOG_TYPE=0)  # Use non-ROS logging
target_link_libraries(autospeed_backend
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBRARY}
    ${TENSORRT_ONNX_LIBRARY}
)

# GStreamer engine
add_library(gstreamer_engine
    ../../common/sensors/gstreamer_engine.cpp
)
target_compile_definitions(gstreamer_engine PRIVATE LOG_TYPE=0)  # Use non-ROS logging
target_link_libraries(gstreamer_engine
    ${OpenCV_LIBS}
    ${GSTREAMER_LIBRARIES}
    ${GSTREAMER_APP_LIBRARIES}
)

# FPS Timer
add_library(fps_timer
    ../../common/benchmark/fps_timer.cpp
)
target_compile_definitions(fps_timer PRIVATE LOG_TYPE=0)  # Use non-ROS logging

# Main executable
add_executable(autospeed_infer_stream
    infer_stream.cpp
)
target_compile_definitions(autospeed_infer_stream PRIVATE LOG_TYPE=0)  # Use non-ROS logging
target_link_libraries(autospeed_infer_stream
    autospeed_backend
    gstreamer_engine
    fps_timer
    ${OpenCV_LIBS}
    pthread
)

# Installation
install(TARGETS autospeed_infer_stream
    RUNTIME DESTINATION bin
)

message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "CUDA: ${CUDA_VERSION}")
message(STATUS "TensorRT: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "GStreamer: ${GSTREAMER_VERSION}")

