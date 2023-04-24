add_executable(phase_field_crystal_simulation_mpi
    phase_field_crystal_simulation_mpi.cpp)

target_link_libraries(phase_field_crystal_simulation_mpi
    PRIVATE
    PhaseFieldCrystalSystemMPI)

install(TARGETS phase_field_crystal_simulation_mpi DESTINATION bin)
