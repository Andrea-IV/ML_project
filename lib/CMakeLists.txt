cmake_minimum_required(VERSION 3.14)
project(lib)

set(CMAKE_CXX_STANDARD 14)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_library(lib SHARED library.cpp library.h)
add_executable(app library.cpp library.h)

target_link_libraries (lib Eigen3::Eigen)
target_link_libraries (app Eigen3::Eigen)