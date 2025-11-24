cmake_minimum_required(VERSION 3.16)
project(autosteer_pipeline)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# ONNX Runtime - Manual configuration
# Set ONNXRUNTIME_ROOT to your ONNX Runtime installation directory
# Example: cmake -DONNXRUNTIME_ROOT=/usr/local/onnxruntime ..
if(NOT DEFINED ONNXRUNTIME_ROOT)
  set(ONNXRUNTIME_ROOT "/usr/local/onnxruntime" CACHE PATH "ONNX Runtime installation directory")
endif()

set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")

# Find ONNX Runtime library
find_library(ONNXRUNTIME_LIBRARY
  NAMES onnxruntime
  PATHS ${ONNXRUNTIME_LIB_DIR}
  NO_DEFAULT_PATH
)

if(NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR "ONNX Runtime library not found. Please set ONNXRUNTIME_ROOT to your ONNX Runtime installation directory.")
endif()

message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIRS}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")

# ===== Inference Library =====
add_library(autosteer_inference
  src/inference/onnxruntime_session.cpp
  src/inference/onnxruntime_engine.cpp
)

target_include_directories(autosteer_inference PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${ONNXRUNTIME_INCLUDE_DIRS}
)

target_link_libraries(autosteer_inference
  ${OpenCV_LIBS}
  ${ONNXRUNTIME_LIBRARY}
)

# ===== Visualization Library =====
add_library(autosteer_visualization
  src/visualization/draw_lanes.cpp
)

target_include_directories(autosteer_visualization PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(autosteer_visualization
  autosteer_inference
  ${OpenCV_LIBS}
)

# ===== Main Executable =====
add_executable(autosteer_pipeline
  main.cpp
)

target_link_libraries(autosteer_pipeline
  autosteer_inference
  autosteer_visualization
  Threads::Threads
  ${OpenCV_LIBS}
)

# Installation
install(TARGETS autosteer_inference autosteer_visualization autosteer_pipeline
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(FILES
  src/inference/onnxruntime_session.hpp
  src/inference/onnxruntime_engine.hpp
  DESTINATION include/autosteer/inference
)

install(FILES
  src/visualization/draw_lanes.hpp
  DESTINATION include/autosteer/visualization
)

# Print configuration summary
message(STATUS "")
message(STATUS "AutoSteer Pipeline Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenCV version: ${OpenCV_VERSION}")
message(STATUS "  ONNX Runtime: ${ONNXRUNTIME_ROOT}")
message(STATUS "  Components:")
message(STATUS "    - Inference library: autosteer_inference")
message(STATUS "    - Visualization library: autosteer_visualization")
message(STATUS "    - Main executable: autosteer_pipeline")
message(STATUS "")

