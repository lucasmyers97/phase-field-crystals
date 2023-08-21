add_library(GridTools
    grid_tools.cpp
    )

target_include_directories(GridTools
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

target_include_directories(GridTools
    PUBLIC
    ${SRC_DIR}
    )

target_link_libraries(GridTools
    PUBLIC
    deal.II::Interface
    )

install(TARGETS GridTools LIBRARY DESTINATION lib)
