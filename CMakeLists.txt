# Basic cmake project commands
cmake_minimum_required(VERSION 3.12)
project(phase-field-crystals VERSION 0.1 LANGUAGES C CXX)

# Must have at least C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# If user doesn't set build type, default to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# In Debug mode, show warnings and produce compile commands for LSP
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-Wall -Wextra -pedantic)
    set(CMAKE_EXPORT_COMPILE_COMMANDS "ON" CACHE STRING "Compile commands" FORCE)
endif()

# source directory so we may reference include files
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# sets runpath so installed binaries know where internal libraries are
file(RELATIVE_PATH relativeRpath
    ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}
    ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}
)
set(CMAKE_INSTALL_RPATH $ORIGIN $ORIGIN/${relativeRpath})

# sets runpath so installed binaries know where external libraries are
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# find Deal II
find_package(deal.II 9.3.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  REQUIRED
  )

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

add_subdirectory(src)
add_subdirectory(app)
