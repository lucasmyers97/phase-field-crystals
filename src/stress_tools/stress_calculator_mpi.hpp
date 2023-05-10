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
    void calculate_stress(const dealii::DoFHandler<dim> &dof_handler,
                          const dealii::LinearAlgebraTrilinos::MPI::BlockVector &Psi);
    void setup_dofs(const MPI_Comm& mpi_communicator);

private:
    dealii::DoFHandler<dim> dof_handler;
    dealii::FESystem<dim> fe_system;
    dealii::AffineConstraints<double> constraints;

    dealii::LinearAlgebraTrilinos::MPI::BlockSparseMatrix system_matrix;
    dealii::LinearAlgebraTrilinos::MPI::BlockVector system_rhs;
    dealii::LinearAlgebraTrilinos::MPI::BlockVector sigma;

    std::vector<dealii::IndexSet> owned_partitioning;
    std::vector<dealii::IndexSet> relevant_partitioning;
};

#endif
