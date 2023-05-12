#ifndef STRESS_CALCULATOR_MPI_HPP
#define STRESS_CALCULATOR_MPI_HPP

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/base/mpi.h>

template <int dim>
class StressCalculatorMPI
{
public:
    StressCalculatorMPI(const dealii::Triangulation<dim> &tria,
                        const unsigned int degree);
    std::vector<unsigned int> calculate_stress(const MPI_Comm& mpi_communicator,
                                               const dealii::DoFHandler<dim> &Psi_dof_handler,
                                               const dealii::LinearAlgebraTrilinos::MPI::BlockVector &Psi,
                                               double eps);
    void setup_dofs(const MPI_Comm& mpi_communicator);
    void calculate_mass_matrix();

private:
    void calculate_righthand_side(const dealii::DoFHandler<dim> &Psi_dof_handler,
                                  const dealii::LinearAlgebraTrilinos::MPI::BlockVector &Psi,
                                  double eps);
    std::vector<unsigned int> solve(const MPI_Comm& mpi_communicator);

    dealii::DoFHandler<dim> dof_handler;
    dealii::FESystem<dim> fe_system;
    dealii::AffineConstraints<double> constraints;

    dealii::LinearAlgebraTrilinos::MPI::BlockSparseMatrix system_matrix;
    dealii::LinearAlgebraTrilinos::MPI::BlockVector system_rhs;
    dealii::LinearAlgebraTrilinos::MPI::BlockVector stress;

    std::vector<dealii::IndexSet> owned_partitioning;
    std::vector<dealii::IndexSet> relevant_partitioning;
};

#endif
