add_library(VectorConversion
    vector_conversion.cpp
    )

target_include_directories(VectorConversion
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

target_include_directories(VectorConversion
    PUBLIC
    ${SRC_DIR}
    )

target_link_libraries(VectorConversion
    PUBLIC
    deal.II::Interface
    )

install(TARGETS VectorConversion LIBRARY DESTINATION lib)
