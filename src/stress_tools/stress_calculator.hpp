#ifndef STRESS_CALCULATOR_HPP
#define STRESS_CALCULATOR_HPP

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/base/index_set.h>

template <bool mpi_enabled>
struct StressCalculatorInternals;

template <>
struct StressCalculatorInternals<true>
{
    using VectorType = dealii::LinearAlgebraTrilinos::MPI::BlockVector;
    using MatrixType = dealii::LinearAlgebraTrilinos::MPI::BlockSparseMatrix;
    using IndexType = std::vector<dealii::IndexSet>;
};



template <>
struct StressCalculatorInternals<false>
{
    using VectorType = dealii::BlockVector<double>;
    using MatrixType = dealii::BlockSparseMatrix<double>;
    using IndexType = dealii::IndexSet;
};

template <int dim, bool mpi_enabled>
class StressCalculator
{
public:
    StressCalculator(const dealii::Triangulation<dim> &tria,
                     const unsigned int degree);
    void setup_dofs(const MPI_Comm& mpi_communicator);
    void calculate_stress(const dealii::DoFHandler<dim> &dof_handler,
                          const typename StressCalculatorInternals<mpi_enabled>::VectorType &Psi);

private:
    dealii::DoFHandler<dim> dof_handler;
    dealii::FESystem<dim> fe_system;
    dealii::AffineConstraints<double> constraints;

    typename StressCalculatorInternals<mpi_enabled>::IndexType locally_owned_dofs;
    typename StressCalculatorInternals<mpi_enabled>::IndexType locally_relevant_dofs;

    typename StressCalculatorInternals<mpi_enabled>::MatrixType system_matrix;
    typename StressCalculatorInternals<mpi_enabled>::VectorType sigma;
};

#endif
