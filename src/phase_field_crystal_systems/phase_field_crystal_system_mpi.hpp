#ifndef PHASE_FIELD_CRYSTAL_SYSTEM_MPI_HPP
#define PHASE_FIELD_CRYSTAL_SYSTEM_MPI_HPP

#include "stress_tools/stress_calculator_mpi.hpp"
#include <deal.II/base/mpi.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/grid/tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/generic_linear_algebra.h>

#include <memory>
#include <utility>

template <int dim>
class PhaseFieldCrystalSystemMPI
{
public:
    PhaseFieldCrystalSystemMPI(unsigned int degree,

                               double eps,

                               double dt,
                               double theta,
                               double simulation_tol,
                               unsigned int simulation_max_iters,

                               unsigned int n_refines,
                               const dealii::Point<dim> &lower_left,
                               const dealii::Point<dim> &upper_right,

                               std::unique_ptr<dealii::Function<dim>> initial_condition
                               );
    void run();

private:
    MPI_Comm mpi_communicator;

    dealii::parallel::distributed::Triangulation<dim> triangulation;
    dealii::FESystem<dim> fe_system;
    dealii::DoFHandler<dim> dof_handler;
    
    std::vector<dealii::IndexSet> owned_partitioning;
    std::vector<dealii::IndexSet> relevant_partitioning;

    dealii::AffineConstraints<double> constraints;
    dealii::BlockSparsityPattern sparsity_pattern;
    dealii::LinearAlgebraTrilinos::MPI::BlockSparseMatrix system_matrix;
    dealii::LinearAlgebraTrilinos::MPI::BlockSparseMatrix M_psi_matrix;
    dealii::LinearAlgebraTrilinos::MPI::BlockVector system_rhs;

    dealii::LinearAlgebraTrilinos::MPI::BlockVector Psi_n;
    dealii::LinearAlgebraTrilinos::MPI::BlockVector Psi_n_1;

    dealii::ConditionalOStream pcout;
    dealii::TimerOutput timer;

    double eps = -0.8;

    double dt = 0.1;
    double theta = 1.0;
    double simulation_tol = 1e-8;
    unsigned int simulation_max_iters = 200;

    unsigned int n_refines = 6;
    dealii::Point<dim> lower_left = {-4 * M_PI / std::sqrt(3), -4 * M_PI / std::sqrt(3)};
    dealii::Point<dim> upper_right = {4 * M_PI / std::sqrt(3), 4 * M_PI / std::sqrt(3)};

    std::unique_ptr<dealii::Function<dim>> initial_condition;

    std::unique_ptr<StressCalculatorMPI<dim>> stress_calculator;

    void make_grid();
    void setup_dofs();
    void initialize_fe_field();
    void assemble_system();
    void solve_and_update();
    void solve_and_update_no_precondition();
    void iterate_timestep();
    void output_configuration(unsigned int iteration);
    void output_rhs(unsigned int iteration);
};

#endif
