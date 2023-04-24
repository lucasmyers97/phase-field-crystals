add_library(PhaseFieldCrystalSystemMPI
    phase_field_crystal_system_mpi.cpp
    )

target_include_directories(PhaseFieldCrystalSystemMPI
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

target_include_directories(PhaseFieldCrystalSystemMPI
    PUBLIC
    ${SRC_DIR}
    )

target_link_libraries(PhaseFieldCrystalSystemMPI
    PUBLIC
    deal.II::Interface
    PhaseFieldFunctions
    )

install(TARGETS PhaseFieldCrystalSystemMPI LIBRARY DESTINATION lib)
