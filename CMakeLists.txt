# Set the minimum version of CMake that's required
cmake_minimum_required(VERSION 3.12)

project(phase-field-crystals VERSION 0.1 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CMAKE_CXX_COMPILER "clang++-13")
set(CMAKE_CXX_COMPILER "g++" CACHE STRING "Compiler")
add_compile_options(-Wall -Wextra -pedantic)
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type")

# find Deal II
find_package(deal.II 9.3.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  REQUIRED)

# massage Deal II into a more modern CMake Target
if(NOT TARGET deal.II::Interface)
  add_library(deal.II::Interface INTERFACE IMPORTED GLOBAL)
  if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    set_target_properties(deal.II::Interface PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${DEAL_II_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${DEAL_II_LIBRARIES_DEBUG}"
      )
  else()
    set_target_properties(deal.II::Interface PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${DEAL_II_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${DEAL_II_LIBRARIES_RELEASE}"
      )
  endif()
endif()

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_subdirectory(src)
add_subdirectory(app)
