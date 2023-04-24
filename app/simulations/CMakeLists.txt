add_executable(phase_field_crystal_mpi_simulation
    phase_field_crystal_mpi_simulation.cpp)

target_link_libraries(phase_field_crystal_mpi_simulation
    PRIVATE
    PhaseFieldCrystalSystemMPI)

install(TARGETS phase_field_crystal_mpi_simulation DESTINATION bin)
