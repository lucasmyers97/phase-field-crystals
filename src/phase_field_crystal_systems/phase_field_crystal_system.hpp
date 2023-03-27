#ifndef PHASE_FIELD_CRYSTAL_SYSTEM
#define PHASE_FIELD_CRYSTAL_SYSTEM

#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/affine_constraints.h>

template <int dim>
class PhaseFieldCrystalSystem
{
public:
    PhaseFieldCrystalSystem(unsigned int degree);
    void run(unsigned int n_refines);

private:
    dealii::Triangulation<dim> triangulation;
    dealii::FESystem<dim> fe_system;
    dealii::DoFHandler<dim> dof_handler;
    dealii::AffineConstraints<double> constraints;

    dealii::BlockVector<double> psi_n;

    void make_grid(unsigned int n_refines);
    void setup_dofs();
    void initialize_fe_field();
    void output_configuration();
};

#endif
