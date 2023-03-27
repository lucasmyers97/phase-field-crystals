#include "phase_field_crystal_systems/phase_field_crystal_system.hpp"
#include "phase_field_functions/hexagonal_lattice.hpp"

int main()
{
    constexpr int dim = 2;
    constexpr unsigned int degree = 1;
    constexpr unsigned int n_refines = 4;

    PhaseFieldCrystalSystem<dim> pfc_system(degree);
    pfc_system.run(n_refines);
    
    return 0;
}
