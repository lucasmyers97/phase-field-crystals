#ifndef STRESS_CALCULATOR_HPP
#define STRESS_CALCULATOR_HPP

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/affine_constraints.h>

template <int dim, typename BlockVector, typename BlockMatrix>
class StressCalculator
{
public:
    StressCalculator(const dealii::Triangulation<dim> &tria,
                     const unsigned int degree);
    void calculate_stress(const dealii::DoFHandler<dim> &dof_handler,
                          const BlockVector &Psi);

private:
    dealii::DoFHandler<dim> dof_handler;
    dealii::FESystem<dim> fe_system;
    dealii::AffineConstraints<double> constraints;

    BlockMatrix system_matrix;
    BlockVector sigma;
};

#endif
