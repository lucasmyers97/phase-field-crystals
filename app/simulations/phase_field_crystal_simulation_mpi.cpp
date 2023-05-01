#include "phase_field_crystal_systems/phase_field_crystal_system_mpi.hpp"

#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

int main(int ac, char* av[])
{
    constexpr int dim = 2;
    constexpr unsigned int degree = 1;

    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(ac, av, 1);

    PhaseFieldCrystalSystemMPI<dim> phase_field_crystal_system_mpi(degree);
    phase_field_crystal_system_mpi.run();

    return 0;
}
