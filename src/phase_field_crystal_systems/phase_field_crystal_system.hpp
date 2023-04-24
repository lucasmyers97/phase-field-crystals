#ifndef PHASE_FIELD_CRYSTAL_SYSTEM_HPP
#define PHASE_FIELD_CRYSTAL_SYSTEM_HPP

#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
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
    dealii::BlockSparsityPattern sparsity_pattern;
    dealii::BlockSparseMatrix<double> system_matrix;
    dealii::BlockVector<double> system_rhs;
    dealii::BlockVector<double> dPsi_n;

    dealii::BlockVector<double> Psi_n;
    dealii::BlockVector<double> Psi_n_1;

    double dt = 0.1;
    double theta = 0.5;
    double eps = -0.8;

    double simulation_tol = 1e-8;
    double simulation_max_iters = 200;

    void make_grid(unsigned int n_refines);
    void setup_dofs();
    void initialize_fe_field();
    void assemble_system();
    void solve_and_update();
    void iterate_timestep();
    void output_configuration(unsigned int iteration);
    void output_rhs(unsigned int iteration);
};

#endif
