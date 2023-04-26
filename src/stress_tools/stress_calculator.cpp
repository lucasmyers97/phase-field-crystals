#include "stress_calculator.hpp"

#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>

template <int dim, bool mpi_enabled>
StressCalculator<dim, mpi_enabled>::
StressCalculator(const dealii::Triangulation<dim> &tria, 
                 const unsigned int degree)
    : dof_handler(tria)
    , fe_system(dealii::FE_Q<dim>(degree)^(dim*dim))
{}



template<int dim>
StressCalculator<dim, true>



template class StressCalculator<2, true>;
template class StressCalculator<2, false>;
