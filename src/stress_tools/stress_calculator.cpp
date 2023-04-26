#include "stress_calculator.hpp"

#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>

template <int dim, typename BlockVector, typename BlockMatrix>
StressCalculator<dim, BlockVector, BlockMatrix>::
StressCalculator(const dealii::Triangulation<dim> &tria, 
                 const unsigned int degree)
    : dof_handler(tria)
    , fe_system(dealii::FE_Q<dim>(degree)^(dim*dim))
{}


template class StressCalculator<2, 
                                dealii::LinearAlgebraTrilinos::MPI::BlockVector, 
                                dealii::LinearAlgebraTrilinos::MPI::BlockSparseMatrix>;
template class StressCalculator<2, 
                                dealii::BlockVector<double>, 
                                dealii::BlockSparseMatrix<double>>;
