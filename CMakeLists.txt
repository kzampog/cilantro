cmake_minimum_required(VERSION 2.8)
project(cilantro)

# Build setup

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

set(CMAKE_BUILD_TYPE "Release")
add_definitions("-O3 -msse2 -msse3")
set(CMAKE_CXX_FLAGS "-std=c++11")

# Packages

find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(include)
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${Pangolin_INCLUDE_DIRS})

# Build library

file(GLOB_RECURSE 3rd_src src/3rd_party/*.c src/3rd_party/*.cpp)
file(GLOB lib_src src/*.cpp)

add_library(cilantro SHARED ${3rd_src} ${lib_src})
target_link_libraries(cilantro ${Pangolin_LIBRARIES})

# Build examples

option(BUILD_EXAMPLES "Build small example apps" ON)
if(BUILD_EXAMPLES)
    file(GLOB example_files RELATIVE ${CMAKE_SOURCE_DIR} examples/*.cpp)
    foreach(example_file ${example_files})
        get_filename_component(example_name ${example_file} NAME_WE)
        add_executable(${example_name} ${example_file})
        target_link_libraries(${example_name} cilantro ${Pangolin_LIBRARIES})
    endforeach(example_file ${example_files})
endif()

# Install target

install(DIRECTORY include/ DESTINATION include)
install(TARGETS cilantro LIBRARY DESTINATION lib)

# Uninstall target

configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY)

add_custom_target(uninstall COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
