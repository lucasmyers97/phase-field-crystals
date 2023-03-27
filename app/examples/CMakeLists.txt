add_executable(initialize_hexagonal_lattice initialize_hexagonal_lattice.cpp)

target_link_libraries(initialize_hexagonal_lattice
    PRIVATE
    PhaseFieldFunctions
    PhaseFieldCrystalSystem)

install(TARGETS initialize_hexagonal_lattice DESTINATION bin/examples)
