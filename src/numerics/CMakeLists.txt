add_library(LocalIntegration
    local_integration.cpp
    )

target_include_directories(LocalIntegration
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

target_include_directories(LocalIntegration
    PUBLIC
    ${SRC_DIR}
    )

target_link_libraries(LocalIntegration
    PUBLIC
    deal.II::Interface
    )

install(TARGETS LocalIntegration LIBRARY DESTINATION lib)
