add_executable(toml_example toml_example.cpp)
target_link_libraries(toml_example 
    PRIVATE 
    toml
    deal.II::Interface
    VectorConversion)

install(TARGETS toml_example DESTINATION bin/examples)

add_subdirectory(local_integration)
