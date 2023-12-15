# Created by the script cgal_create_CMakeLists
# This is the CMake script for compiling a set of CGAL applications.

cmake_minimum_required(VERSION 3.1...3.23)

project( GravoMG_Visualisation )


# CGAL and its components
find_package( CGAL QUIET COMPONENTS  )

if ( NOT CGAL_FOUND )

  message(STATUS "This project requires the CGAL library, and will not be compiled.")
  return()  

endif()


# OpenGL
find_package(OpenGL REQUIRED)

if ( NOT OPENGL_FOUND )

  message(STATUS "This project requires the OpenGL library, and will not be compiled.")

  return()

endif()


# GLFW
find_package(glfw3 REQUIRED)

# GLEW
find_package(GLEW REQUIRED)

# GLM
find_package(glm REQUIRED)

# Eigen
find_package (Eigen3 3.3 REQUIRED NO_MODULE)


# Creating entries for target: GravoMG_Visualisation
# ##################################################

add_executable(GravoMG_Visualisation main.cpp)

add_to_cached_list(CGAL_EXECUTABLE_TARGETS GravoMG_Visualisation)

target_include_directories(GravoMG_Visualisation PUBLIC ${OPENGL_INCLUDE_DIR})

# Link the executable to CGAL and third-party libraries
target_link_libraries(GravoMG_Visualisation PRIVATE CGAL::CGAL)

target_link_libraries(GravoMG_Visualisation PUBLIC glfw ${OPENGL_gl_LIBRARY})
target_link_libraries(GravoMG_Visualisation PUBLIC GLEW::GLEW)
target_link_libraries(GravoMG_Visualisation PUBLIC glm::glm)
target_link_libraries(GravoMG_Visualisation PUBLIC Eigen3::Eigen)

# ImGui
set(IMGUI_DIR imgui)
file(GLOB IMGUI_SOURCES ${IMGUI_DIR}/*.cpp)
target_sources(GravoMG_Visualisation PUBLIC ${IMGUI_SOURCES})

# NFD
add_subdirectory(nativefiledialog-extended)
target_link_libraries(GravoMG_Visualisation PUBLIC nfd)

# GravoMG
add_subdirectory(gravomg)
target_link_libraries(GravoMG_Visualisation PUBLIC gravomg)
