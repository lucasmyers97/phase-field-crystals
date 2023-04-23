#include "phase_field_crystal_systems/phase_field_crystal_system.hpp"

#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

int main(int ac, char* av[])
{
    constexpr int dim = 2;
    constexpr unsigned int degree = 1;
    constexpr unsigned int n_refines = 8;

    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(ac, av, 1);

    PhaseFieldCrystalSystem<dim> phase_field_crystal_system(degree);
    phase_field_crystal_system.run(n_refines);

    return 0;
}
