cmake_minimum_required(VERSION 2.8)
project(sisyphus)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# compile with C++11 support
add_definitions("-std=c++11")

find_package(Pangolin REQUIRED)

include_directories(${Pangolin_INCLUDE_DIRS})
include_directories(include)

file(GLOB src_files src/*.cpp)

set(LIB_NAME "sisyphus")

add_library(${LIB_NAME} SHARED ${src_files})
target_link_libraries(${LIB_NAME} ${Pangolin_LIBRARIES})

install(TARGETS ${LIB_NAME} LIBRARY DESTINATION /usr/local/lib)
install(DIRECTORY include/ DESTINATION /usr/local/include)

# uninstall target
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
	"${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
	IMMEDIATE @ONLY)

add_custom_target(uninstall COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
