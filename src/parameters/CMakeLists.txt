# use TOML library from submodule
add_library(toml toml.cpp)

target_include_directories(toml PUBLIC ${CMAKE_SOURCE_DIR})
target_include_directories(toml PUBLIC ${SRC_DIR})

install(TARGETS toml LIBRARY DESTINATION lib)
