add_executable(phase_field_crystal_simulation
    phase_field_crystal_simulation.cpp
    )
target_link_libraries(phase_field_crystal_simulation
    PRIVATE
    PhaseFieldCrystalSystem
    PhaseFieldFunctions
    )

add_executable(phase_field_crystal_simulation_mpi
    phase_field_crystal_simulation_mpi.cpp
    )
target_link_libraries(phase_field_crystal_simulation_mpi
    PRIVATE
    PhaseFieldCrystalSystemMPI
    )

install(TARGETS phase_field_crystal_simulation DESTINATION bin)
install(TARGETS phase_field_crystal_simulation_mpi DESTINATION bin)
