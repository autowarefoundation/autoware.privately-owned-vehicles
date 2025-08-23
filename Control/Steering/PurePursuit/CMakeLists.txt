cmake_minimum_required(VERSION 3.10)
project(PurePursuit)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

add_executable(PurePursuit
    src/pure_pursuit.cpp
)
