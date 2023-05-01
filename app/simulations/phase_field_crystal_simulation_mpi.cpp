#include "phase_field_crystal_systems/phase_field_crystal_system_mpi.hpp"

#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <memory>
#include <utility>

#include "phase_field_functions/hexagonal_lattice.hpp"

int main(int ac, char* av[])
{
    constexpr int dim = 2;
    constexpr unsigned int degree = 1;

    double eps = -0.8;

    double dt = 0.1;
    double theta = 1.0;
    double simulation_tol = 1e-8;
    unsigned int simulation_max_iters = 200;

    unsigned int n_refines = 6;

    const double a = 4 * M_PI / std::sqrt(3);

    const double left = -6 * a;
    const double down = -5 * a;

    dealii::Point<dim> p1 = {left, down};
    dealii::Point<dim> p2 = -p1;

    double psi_0 = -0.43;
    double A_0 = 0.2 * (std::abs(psi_0)
                        + (1.0 / 3.0) * std::sqrt(-15 * eps - 36 * psi_0 * psi_0));

    std::vector<dealii::Tensor<1, dim>> dislocation_positions;
    // dislocation_positions.push_back(dealii::Tensor<1, dim>({2 * a, 0}));
    // dislocation_positions.push_back(dealii::Tensor<1, dim>({-2 * a, 0}));

    std::vector<dealii::Tensor<1, dim>> burgers_vectors;
    // burgers_vectors.push_back(dealii::Tensor<1, dim>({a, 0}));
    // burgers_vectors.push_back(dealii::Tensor<1, dim>({-a, 0}));

    std::unique_ptr<dealii::Function<dim>> initial_condition 
        = std::make_unique<HexagonalLattice<dim>>(A_0, 
                                                  psi_0, 
                                                  dislocation_positions, 
                                                  burgers_vectors);

    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(ac, av, 1);

    PhaseFieldCrystalSystemMPI<dim> phase_field_crystal_system_mpi(degree,
                                                                   eps,
                                                                   dt,
                                                                   theta,
                                                                   simulation_tol,
                                                                   simulation_max_iters,
                                                                   n_refines,
                                                                   p1,
                                                                   p2,
                                                                   std::move(initial_condition));
    phase_field_crystal_system_mpi.run();

    return 0;
}
