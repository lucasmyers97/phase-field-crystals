#include "phase_field_crystal_systems/phase_field_crystal_system.hpp"

int main()
{
    constexpr int dim = 2;
    constexpr unsigned int n_refines = 6;

    PhaseFieldCrystalSystem<dim> phase_field_crystal_system;
    phase_field_crystal_system.run(n_refines);

    return 0;
}
