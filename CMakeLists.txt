cmake_minimum_required(VERSION 2.8)
project(cilantro)

set(LIBRARY_NAME ${PROJECT_NAME})

# Build setup

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

set(CMAKE_BUILD_TYPE "Release")
add_definitions("-O3 -msse2 -msse3")
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11")

# Packages

find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(OTHER_INCLUDES ${EIGEN3_INCLUDE_DIRS} ${Pangolin_INCLUDE_DIRS})

include_directories(include)
include_directories(${OTHER_INCLUDES})

# Build library

file(GLOB_RECURSE 3rd_src ${CMAKE_SOURCE_DIR}/src/3rd_party/*.c ${CMAKE_SOURCE_DIR}/src/3rd_party/*.cpp)
file(GLOB lib_src ${CMAKE_SOURCE_DIR}/src/*.cpp)

add_library(${LIBRARY_NAME} SHARED ${3rd_src} ${lib_src})
target_link_libraries(${LIBRARY_NAME} ${Pangolin_LIBRARIES})

# Build examples

option(BUILD_EXAMPLES "Build small example apps" ON)
if(BUILD_EXAMPLES)
    file(GLOB example_files RELATIVE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/examples/*.cpp)
    foreach(example_file ${example_files})
        get_filename_component(example_name ${example_file} NAME_WE)
        add_executable(${example_name} ${example_file})
        target_link_libraries(${example_name} ${LIBRARY_NAME} ${Pangolin_LIBRARIES})
    endforeach(example_file ${example_files})
endif()

# Package config files

file(REMOVE "${CMAKE_BINARY_DIR}/${LIBRARY_NAME}Targets.cmake")
export(TARGETS ${LIBRARY_NAME} APPEND FILE "${CMAKE_BINARY_DIR}/${LIBRARY_NAME}Targets.cmake")

set(CMAKECONFIG_INSTALL_DIR lib/cmake/${LIBRARY_NAME})
file(RELATIVE_PATH REL_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/${CMAKECONFIG_INSTALL_DIR}" "${CMAKE_INSTALL_PREFIX}/include")

# Build tree config
set(EXPORT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
configure_file(${CMAKE_SOURCE_DIR}/${LIBRARY_NAME}Config.cmake.in ${CMAKE_BINARY_DIR}/${LIBRARY_NAME}Config.cmake @ONLY IMMEDIATE)

# Install tree config
set(EXPORT_INCLUDE_DIR "\${PROJECT_CMAKE_DIR}/${REL_INCLUDE_DIR}")
configure_file(${CMAKE_SOURCE_DIR}/${LIBRARY_NAME}Config.cmake.in ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${LIBRARY_NAME}Config.cmake @ONLY)

# Add package to CMake package registery for use from the build tree
option(EXPORT_${LIBRARY_NAME} "Export ${LIBRARY_NAME} package for use by other software" ON)
if(EXPORT_${LIBRARY_NAME})
    export(PACKAGE ${LIBRARY_NAME})
endif()

# Install target

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
install(TARGETS ${LIBRARY_NAME} EXPORT ${PROJECT_NAME}Targets LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

install(FILES "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${LIBRARY_NAME}Config.cmake" DESTINATION ${CMAKECONFIG_INSTALL_DIR})
install(EXPORT ${LIBRARY_NAME}Targets DESTINATION ${CMAKECONFIG_INSTALL_DIR})

# Uninstall target

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake" IMMEDIATE @ONLY)
add_custom_target(uninstall COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
