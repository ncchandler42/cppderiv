cmake_minimum_required(VERSION 3.10)

project(cppderiv VERSION "0.1.0")

enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)
add_compile_options(-Wall -O3 -g)

find_package(Eigen3 REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(cppderiv src/cppderiv.cpp src/bind.cpp)
target_link_libraries(cppderiv PRIVATE Eigen3::Eigen)

option(DEBUG "Enable debugging symbols" OFF)
option(COMPILE_TESTS "Compile proof-of-concept tests" OFF)

if(DEBUG)
	add_definitions(-DDEBUG)
endif()

if(COMPILE_TESTS)
	add_executable(eigen_expr tests/eigen_expr.cpp)
	target_link_libraries(eigen_expr PRIVATE Eigen3::Eigen)

	add_executable(fix_indices tests/fix_indices.cpp)
endif()