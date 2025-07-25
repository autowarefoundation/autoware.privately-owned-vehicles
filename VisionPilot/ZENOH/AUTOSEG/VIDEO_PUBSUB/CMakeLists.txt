cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(video_pubsub)

## Find libraries
find_package(OpenCV REQUIRED HINTS ${OPENCV_INSTALL_ROOT}/build/ QUIET)
find_package(zenohc REQUIRED)

string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")
string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")

## ONNX Runtime Providers
option(LIBPNG_ROOTDIR "libpng root dir")
set(CMAKE_CXX_STANDARD 17)

include_directories(
                     ${OpenCV_INCLUDE_DIRS}    # OpenCV
                   )

## Build & Link
add_executable(video_publisher video_publisher.cpp)
target_link_libraries(video_publisher PRIVATE ${OpenCV_LIBS} opencv_videoio opencv_imgproc opencv_highgui zenohc::lib)
add_executable(video_subscriber video_subscriber.cpp)
target_link_libraries(video_subscriber PRIVATE ${OpenCV_LIBS} opencv_videoio opencv_imgproc opencv_highgui zenohc::lib)
