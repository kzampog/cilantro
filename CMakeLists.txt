cmake_minimum_required(VERSION 3.10)
project(cilantro)

set(CMAKE_BUILD_TYPE "Release")

# Find dependencies
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenMP)
find_package(Pangolin QUIET)

# Library build options
if(WIN32)
  option(BUILD_SHARED_LIBS "Build shared library" OFF)
else()
  option(BUILD_SHARED_LIBS "Build shared library" ON)
endif()
option(ENABLE_NATIVE_BUILD_OPTIMIZATIONS "Enable native build optimization flags" ON)
option(
  ENABLE_NON_DETERMINISTIC_PARALLELISM
  "Enable parallelization that may produce non-deterministic results"
  ON
)

# Find library header/source files
file(
  GLOB_RECURSE
  LIB_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}/*.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}/*.hpp
  ${CMAKE_CURRENT_BINARY_DIR}/include/${PROJECT_NAME}/*.h
  ${CMAKE_CURRENT_BINARY_DIR}/include/${PROJECT_NAME}/*.hpp
)
file(
  GLOB_RECURSE
  LIB_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

# Add library
add_library(${PROJECT_NAME} ${LIB_HEADERS} ${LIB_SOURCES})
target_include_directories(
  ${PROJECT_NAME}
  SYSTEM
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
  target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
  set(DEPENDS_OpenMP "find_dependency(OpenMP)")
endif()
if(Pangolin_FOUND)
  target_link_libraries(${PROJECT_NAME} PUBLIC ${Pangolin_LIBRARIES})
  set(HAVE_PANGOLIN ON)
  set(DEPENDS_Pangolin "find_dependency(Pangolin)")
endif()
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
if(ENABLE_NATIVE_BUILD_OPTIMIZATIONS)
  target_compile_options(${PROJECT_NAME} PUBLIC -O3 -march=native -mtune=native)
else()
  target_compile_options(${PROJECT_NAME} PUBLIC -O3)
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

# Configure preprocessor definitions
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.hpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/include/${PROJECT_NAME}/config.hpp"
)

# Package config files
export(
  TARGETS ${PROJECT_NAME}
  FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake"
)

option(EXPORT_${PROJECT_NAME} "Export ${PROJECT_NAME} package for use by other software" ON)
if(EXPORT_${PROJECT_NAME})
  # Add package to CMake package registry for use from the build tree
  export(PACKAGE ${PROJECT_NAME})
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  @ONLY
  IMMEDIATE
)

set(CMAKECONFIG_INSTALL_DIR lib/cmake/${PROJECT_NAME})

# Install headers
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/include/${PROJECT_NAME}/config.hpp"
  DESTINATION include/${PROJECT_NAME}
)
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
  DESTINATION include
)

# Install library
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

# Install package files
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  DESTINATION ${CMAKECONFIG_INSTALL_DIR}
)
install(
  EXPORT ${PROJECT_NAME}Targets
  DESTINATION ${CMAKECONFIG_INSTALL_DIR}
)

# Uninstall target
if(NOT TARGET uninstall)
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY
  )
  add_custom_target(
    uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
  )
endif()

# Configure Doxyfile
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Doxyfile.in
  ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
  @ONLY
)

# Build examples
option(BUILD_EXAMPLES "Build small example apps" ON)
if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
