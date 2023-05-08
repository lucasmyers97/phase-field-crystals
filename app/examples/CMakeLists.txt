add_executable(toml_example toml_example.cpp)
target_link_libraries(toml_example PRIVATE toml)

install(TARGETS toml_example DESTINATION bin/examples)
