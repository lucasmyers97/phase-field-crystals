add_library(PhaseFieldCrystalSystem
    phase_field_crystal_system.cpp
    )

target_include_directories(PhaseFieldCrystalSystem
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

target_include_directories(PhaseFieldCrystalSystem
    PUBLIC
    ${SRC_DIR}
    )

target_link_libraries(PhaseFieldCrystalSystem
    PUBLIC
    deal.II::Interface
    PhaseFieldFunctions
    )

install(TARGETS PhaseFieldCrystalSystem LIBRARY DESTINATION lib)
