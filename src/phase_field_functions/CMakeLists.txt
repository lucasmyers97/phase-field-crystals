add_library(PhaseFieldFunctions
    hexagonal_lattice.cpp
    )

target_include_directories(PhaseFieldFunctions
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

target_include_directories(PhaseFieldFunctions
    PUBLIC
    ${SRC_DIR}
    )

target_link_libraries(PhaseFieldFunctions
    PUBLIC
    deal.II::Interface
    )

install(TARGETS PhaseFieldFunctions LIBRARY DESTINATION lib)
