cmake_minimum_required(VERSION 3.10)
project(PurePursuit)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

add_executable(PurePursuit
    src/main.cpp
    src/pure_pursuit.cpp
)

add_library(pure_pursuit_lib
    src/pure_pursuit.cpp
)
target_include_directories(pure_pursuit_lib PUBLIC include)