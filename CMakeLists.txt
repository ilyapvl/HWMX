cmake_minimum_required(VERSION 3.14)
project(TriangleIntersection)

set(CMAKE_CXX_STANDARD 20)

include_directories(include)

add_executable(determinant_main src/main.cpp src/determinant.cpp)

