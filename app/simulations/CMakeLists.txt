add_executable(phase_field_crystal_simulation phase_field_crystal_simulation.cpp)

target_link_libraries(phase_field_crystal_simulation
    PRIVATE
    PhaseFieldCrystalSystem)

install(TARGETS phase_field_crystal_simulation DESTINATION bin)
