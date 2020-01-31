cmake_minimum_required(VERSION 3.10)

project(cppderiv)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)
add_compile_options(-Wall -O3 -g)

find_package(Eigen3 REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(cppderiv src/bind.cpp)
target_link_libraries(cppderiv PRIVATE Eigen3::Eigen)