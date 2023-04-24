#include "phase_field_crystal_systems/phase_field_crystal_system.hpp"

int main()
{
    constexpr int dim = 2;
    constexpr unsigned int degree = 1;
    constexpr unsigned int n_refines = 5;

    PhaseFieldCrystalSystem<dim> phase_field_crystal_system(degree);
    phase_field_crystal_system.run(n_refines);

    return 0;
}
