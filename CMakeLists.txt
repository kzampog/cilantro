cmake_minimum_required(VERSION 3.9)
project(cilantro)

# Build setup
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 11)

# Packages
find_package(Eigen3 REQUIRED NO_MODULE)
# below is a workaround for some buggy Eigen exports which managed to get into some distros
if(NOT TARGET Eigen3::Eigen)
    message("Eigen has buggy exported package, but headers are found: ${EIGEN3_INCLUDE_DIRS}. Applying workaround.")
    add_library(Eigen INTERFACE IMPORTED GLOBAL)
    target_include_directories(Eigen INTERFACE "${EIGEN3_INCLUDE_DIRS}")
    add_library(Eigen3::Eigen ALIAS Eigen)
endif()

find_package(Pangolin QUIET)
if(Pangolin_FOUND)
    set(HAVE_PANGOLIN ON)
    set(DEPENDS_Pangolin "find_dependency(Pangolin)")
endif()
find_package(OpenMP)
# if(OPENMP_FOUND)
#     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# endif()

set(OTHER_INCLUDE_DIRS ${Pangolin_INCLUDE_DIRS})
set(OTHER_LIBRARIES ${Pangolin_LIBRARIES})

if(_WIN_)
    option(BUILD_SHARED_LIBS "Build Shared Library" OFF)
else()
    option(BUILD_SHARED_LIBS "Build Shared Library" ON)
endif()

# Build options
option(ENABLE_NATIVE_BUILD_OPTIMIZATIONS "Enable native build optimization flags" ON)
option(ENABLE_NON_DETERMINISTIC_PARALLELISM "Enable parallelization that may produce non-deterministic results" ON)

# Build library
file(GLOB_RECURSE THIRD_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/3rd_party/*.c ${CMAKE_CURRENT_SOURCE_DIR}/src/3rd_party/*.cpp)
file(GLOB_RECURSE LIB_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}/*.hpp ${CMAKE_CURRENT_BINARY_DIR}/include/${PROJECT_NAME}/*.hpp)
file(GLOB_RECURSE LIB_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_library(${PROJECT_NAME} ${THIRD_SOURCES} ${LIB_HEADERS} ${LIB_SOURCES})
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${OTHER_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                                                  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
                                                  $<INSTALL_INTERFACE:include>)
target_link_libraries(${PROJECT_NAME} PUBLIC ${OTHER_LIBRARIES} Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
    set(DEPENDS_OpenMP "find_dependency(OpenMP)")
endif()

if (ENABLE_NATIVE_BUILD_OPTIMIZATIONS)
    target_compile_options(${PROJECT_NAME} PUBLIC -O3 -march=native -mtune=native)
else()
    target_compile_options(${PROJECT_NAME} PUBLIC -O3)
endif()

# Configure preprocessor definitions
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/config.hpp.in" "${CMAKE_CURRENT_BINARY_DIR}/include/${PROJECT_NAME}/config.hpp")

# Package config files
file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake")
export(TARGETS ${PROJECT_NAME} APPEND FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake")

set(CMAKECONFIG_INSTALL_DIR lib/cmake/${PROJECT_NAME})
file(RELATIVE_PATH REL_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/${CMAKECONFIG_INSTALL_DIR}" "${CMAKE_INSTALL_PREFIX}/include")

# Build tree config
set(EXPORT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include;${CMAKE_CURRENT_BINARY_DIR}/include")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}Config.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake @ONLY IMMEDIATE)

# Install tree config
set(EXPORT_INCLUDE_DIR "\${PROJECT_CMAKE_DIR}/${REL_INCLUDE_DIR}")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}Config.cmake.in ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${PROJECT_NAME}Config.cmake @ONLY)

# Add package to CMake package registry for use from the build tree
option(EXPORT_${PROJECT_NAME} "Export ${PROJECT_NAME} package for use by other software" ON)
if(EXPORT_${PROJECT_NAME})
    export(PACKAGE ${PROJECT_NAME})
endif()

# Install target
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/${PROJECT_NAME}/config.hpp" DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${PROJECT_NAME}Config.cmake" DESTINATION ${CMAKECONFIG_INSTALL_DIR})
install(EXPORT ${PROJECT_NAME}Targets DESTINATION ${CMAKECONFIG_INSTALL_DIR})

# Uninstall target
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake" IMMEDIATE @ONLY)
add_custom_target(uninstall COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)

# Configure Doxyfile
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)

# Build examples
option(BUILD_EXAMPLES "Build small example apps" ON)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
