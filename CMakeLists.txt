cmake_minimum_required(VERSION 2.8)
project(cilantro)

# Build setup

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

set(CMAKE_BUILD_TYPE "Release")
add_definitions("-O3 -msse2 -msse3")
set(CMAKE_CXX_FLAGS "-std=c++11")

# Packages

find_package(Pangolin REQUIRED)

include_directories(${Pangolin_INCLUDE_DIRS})
include_directories(3rd_party)
include_directories(include)

# Build library

file(GLOB_RECURSE lib_src src/*.cpp)
file(GLOB_RECURSE 3rd_src 3rd_party/*.cpp 3rd_party/*.c)

add_library(cilantro SHARED ${3rd_src} ${lib_src})
target_link_libraries(cilantro ${Pangolin_LIBRARIES})

# Build examples

add_executable(ply_io_example examples/ply_io_example.cpp)
target_link_libraries(ply_io_example cilantro ${Pangolin_LIBRARIES})

add_executable(principal_component_analysis_example examples/principal_component_analysis_example.cpp)
target_link_libraries(principal_component_analysis_example cilantro ${Pangolin_LIBRARIES})

add_executable(voxel_grid_example examples/voxel_grid_example.cpp)
target_link_libraries(voxel_grid_example cilantro ${Pangolin_LIBRARIES})

add_executable(kd_tree_example examples/kd_tree_example.cpp)
target_link_libraries(kd_tree_example cilantro ${Pangolin_LIBRARIES})

add_executable(normal_estimation_example examples/normal_estimation_example.cpp)
target_link_libraries(normal_estimation_example cilantro ${Pangolin_LIBRARIES})

add_executable(matrix_io_example examples/matrix_io_example.cpp)
target_link_libraries(matrix_io_example cilantro ${Pangolin_LIBRARIES})

add_executable(visualizer_example examples/visualizer_example.cpp)
target_link_libraries(visualizer_example cilantro ${Pangolin_LIBRARIES})

add_executable(convex_hull_example examples/convex_hull_example.cpp)
target_link_libraries(convex_hull_example cilantro ${Pangolin_LIBRARIES})

add_executable(connected_component_segmentation_example examples/connected_component_segmentation_example.cpp)
target_link_libraries(connected_component_segmentation_example cilantro ${Pangolin_LIBRARIES})

add_executable(image_viewer_example examples/image_viewer_example.cpp)
target_link_libraries(image_viewer_example cilantro ${Pangolin_LIBRARIES})

add_executable(image_point_cloud_conversions_example examples/image_point_cloud_conversions_example.cpp)
target_link_libraries(image_point_cloud_conversions_example cilantro ${Pangolin_LIBRARIES})

add_executable(iterative_closest_point_example examples/iterative_closest_point_example.cpp)
target_link_libraries(iterative_closest_point_example cilantro ${Pangolin_LIBRARIES})

add_executable(plane_estimator_example examples/plane_estimator_example.cpp)
target_link_libraries(plane_estimator_example cilantro ${Pangolin_LIBRARIES})

add_executable(rigid_transform_estimator_example examples/rigid_transform_estimator_example.cpp)
target_link_libraries(rigid_transform_estimator_example cilantro ${Pangolin_LIBRARIES})

# Install target

install(TARGETS cilantro LIBRARY DESTINATION /usr/local/lib)
install(DIRECTORY include/ DESTINATION /usr/local/include)

# Uninstall target

configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY)

add_custom_target(uninstall COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
