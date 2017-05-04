cmake_minimum_required(VERSION 2.8)
project(sisyphus)

# Build setup

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

set(CMAKE_BUILD_TYPE "Release")
add_definitions("-std=c++11 -O3 -msse2 -msse3")

# Packages

find_package(Pangolin REQUIRED)

include_directories(${Pangolin_INCLUDE_DIRS})
include_directories(include)

# Build library

file(GLOB lib_src src/*.cpp)

add_library(sisyphus SHARED ${lib_src})
target_link_libraries(sisyphus ${Pangolin_LIBRARIES})

# Build examples

add_executable(taka examples/taka.cpp)
target_link_libraries(taka sisyphus ${Pangolin_LIBRARIES})

# Install target

install(TARGETS sisyphus LIBRARY DESTINATION /usr/local/lib)
install(DIRECTORY include/ DESTINATION /usr/local/include)

# Uninstall target
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
	"${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
	IMMEDIATE @ONLY)

add_custom_target(uninstall COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
