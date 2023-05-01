#include "phase_field_crystal_system_mpi.hpp"

#include <deal.II/base/mpi.h>

#include <deal.II/base/array_view.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/types.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/grid/tria.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/numerics/data_component_interpretation.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/trilinos_linear_operator.h>

#include <deal.II/numerics/vector_tools_interpolate.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>
#include <string>

#include "phase_field_functions/hexagonal_lattice.hpp"

class PsiMatrix : public dealii::Subscriptor
{
public:
    PsiMatrix(const dealii::SparseMatrix<double> &B,
              const dealii::SparseMatrix<double> &C,
              const dealii::SparseMatrix<double> &D,
              const dealii::SparseMatrix<double> &L_psi,
              const dealii::SparseMatrix<double> &L_chi,
              const dealii::SparseDirectUMFPACK &M_chi_inv,
              const dealii::SparseDirectUMFPACK &M_phi_inv)
        : B(B)
        , C(C)
        , D(D)
        , L_psi(L_psi)
        , L_chi(L_chi)
        , M_chi_inv(M_chi_inv)
        , M_phi_inv(M_phi_inv)
    {};

    void vmult(dealii::Vector<double> &dst, dealii::Vector<double> &src) const;

private:
    const dealii::SparseMatrix<double> &B;
    const dealii::SparseMatrix<double> &C;
    const dealii::SparseMatrix<double> &D;
    const dealii::SparseMatrix<double> &L_psi;
    const dealii::SparseMatrix<double> &L_chi;
    const dealii::SparseDirectUMFPACK &M_chi_inv;
    const dealii::SparseDirectUMFPACK &M_phi_inv;
};



void PsiMatrix::vmult(dealii::Vector<double> &dst, dealii::Vector<double> &src) const
{
    dealii::Vector<double> tmp_psi(src.size());
    dealii::Vector<double> tmp_chi_1(L_psi.m());
    dealii::Vector<double> tmp_chi_2(L_psi.m());
    dealii::Vector<double> tmp_phi_1(L_chi.m());
    dealii::Vector<double> tmp_phi_2(L_chi.m());

    L_psi.vmult(tmp_chi_1, src);
    M_chi_inv.vmult(tmp_chi_2, tmp_chi_1);

    C.vmult(tmp_psi, tmp_chi_2);

    L_chi.vmult(tmp_phi_1, tmp_chi_2);
    M_phi_inv.vmult(tmp_phi_2, tmp_phi_1);
    D.vmult(dst, tmp_phi_2);

    dst -= tmp_psi;

    B.vmult(tmp_psi, src);

    dst += tmp_psi;
}



template <int dim>
PhaseFieldCrystalSystemMPI<dim>::PhaseFieldCrystalSystemMPI(unsigned int degree)
    : mpi_communicator(MPI_COMM_WORLD)
    , triangulation(mpi_communicator,
                    typename dealii::Triangulation<dim>::MeshSmoothing(
                    dealii::Triangulation<dim>::smoothing_on_refinement |
                    dealii::Triangulation<dim>::smoothing_on_coarsening))
    , fe_system(dealii::FE_Q<dim>(degree), 1,
                dealii::FE_Q<dim>(degree), 1,
                dealii::FE_Q<dim>(degree), 1)
    , dof_handler(triangulation)
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , timer(mpi_communicator,
            pcout,
            dealii::TimerOutput::never,
            dealii::TimerOutput::wall_times)
{
    double psi_0 = -0.43;
    double A_0 = 0.2 * (std::abs(psi_0)
                        + (1.0 / 3.0) * std::sqrt(-15 * eps - 36 * psi_0 * psi_0));

    double a = 4 * M_PI / std::sqrt(3);

    std::vector<dealii::Tensor<1, dim>> dislocation_positions;
    // dislocation_positions.push_back(dealii::Tensor<1, dim>({2 * a, 0}));
    // dislocation_positions.push_back(dealii::Tensor<1, dim>({-2 * a, 0}));

    std::vector<dealii::Tensor<1, dim>> burgers_vectors;
    // burgers_vectors.push_back(dealii::Tensor<1, dim>({a, 0}));
    // burgers_vectors.push_back(dealii::Tensor<1, dim>({-a, 0}));

    initial_condition = std::make_unique<HexagonalLattice<dim>>(A_0, 
                                                                psi_0, 
                                                                dislocation_positions, 
                                                                burgers_vectors);
}



template <int dim>
void PhaseFieldCrystalSystemMPI<dim>::make_grid()
{
    const double a = 4 * M_PI / std::sqrt(3);

    // const int n_across = 5;
    // const int n_down = 5;
    // const double left = -n_across * (2 / std::sqrt(3)) * M_PI * 2;
    // const double down = -n_down * 2 * M_PI;
    // const double left = -20 * a;
    // const double down = -20 * std::sqrt(3) * a;
    const double left = -6 * a;
    const double down = -5 * a;

    dealii::Point<dim> p1 = {left, down};
    dealii::Point<dim> p2 = -p1;

    dealii::GridGenerator::hyper_rectangle(triangulation, 
                                           p1, 
                                           p2, 
                                           /*colorize*/ true);

    using PeriodicFaces
        = std::vector<dealii::GridTools::PeriodicFacePair<
            typename dealii::parallel::distributed::Triangulation<dim>::cell_iterator
                >
            >;

    PeriodicFaces x_periodic_faces;
    PeriodicFaces y_periodic_faces;
    dealii::GridTools::collect_periodic_faces(triangulation,
                                              /*b_id1*/ 0,
                                              /*b_id2*/ 1,
                                              /*direction*/ 0,
                                              x_periodic_faces);
    dealii::GridTools::collect_periodic_faces(triangulation,
                                              /*b_id1*/ 2,
                                              /*b_id2*/ 3,
                                              /*direction*/ 1,
                                              y_periodic_faces);

    triangulation.add_periodicity(x_periodic_faces);
    triangulation.add_periodicity(y_periodic_faces);
    triangulation.refine_global(n_refines);
}



template <int dim>
void PhaseFieldCrystalSystemMPI<dim>::setup_dofs()
{
    dof_handler.distribute_dofs(fe_system);

    std::vector<unsigned int> block_component = {0, 1, 2};
    dealii::DoFRenumbering::component_wise(dof_handler, block_component);

    const std::vector<dealii::types::global_dof_index> 
        dofs_per_block = dealii::DoFTools::count_dofs_per_fe_block(dof_handler, block_component);

    const dealii::types::global_dof_index n_psi = dofs_per_block[0];
    const dealii::types::global_dof_index n_chi = dofs_per_block[1];
    const dealii::types::global_dof_index n_phi = dofs_per_block[2];

    owned_partitioning.resize(3);
    owned_partitioning[0] = dof_handler
                            .locally_owned_dofs()
                            .get_view(0, n_psi);
    owned_partitioning[1] = dof_handler
                            .locally_owned_dofs()
                            .get_view(n_psi, n_psi + n_chi);
    owned_partitioning[2] = dof_handler
                            .locally_owned_dofs()
                            .get_view(n_psi + n_chi, n_psi + n_chi + n_phi);

    const dealii::IndexSet locally_relevant_dofs =
        dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);
    relevant_partitioning.resize(3);
    relevant_partitioning[0] = locally_relevant_dofs
                               .get_view(0, n_psi);
    relevant_partitioning[1] = locally_relevant_dofs
                               .get_view(n_psi, n_psi + n_chi);
    relevant_partitioning[2] = locally_relevant_dofs
                               .get_view(n_psi + n_chi, n_psi + n_chi + n_phi);
    {
        constraints.clear();
        constraints.reinit(locally_relevant_dofs);
        dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);

        using PeriodicFaces
            = std::vector<dealii::GridTools::PeriodicFacePair<
                typename dealii::DoFHandler<dim>::cell_iterator
                    >
                >;

        PeriodicFaces x_periodic_faces;
        PeriodicFaces y_periodic_faces;
        dealii::GridTools::collect_periodic_faces(dof_handler,
                                                  /*b_id1*/ 0,
                                                  /*b_id2*/ 1,
                                                  /*direction*/ 0,
                                                  x_periodic_faces);
        dealii::GridTools::collect_periodic_faces(dof_handler,
                                                  /*b_id1*/ 2,
                                                  /*b_id2*/ 3,
                                                  /*direction*/ 1,
                                                  y_periodic_faces);

        dealii::DoFTools::
            make_periodicity_constraints<dim, dim>(x_periodic_faces,
                                                   constraints);
        dealii::DoFTools::
            make_periodicity_constraints<dim, dim>(y_periodic_faces,
                                                   constraints);
        constraints.close();
    }
    {
        dealii::Table<2, dealii::DoFTools::Coupling> coupling(3, 3);
        for (unsigned int c = 0; c < 3; ++c)
            for (unsigned int d = 0; d < 3; ++d)
            {
                if ( ((c == 2) && (d == 0))
                    ||((c == 1) && (d == 2)) )
                    coupling[c][d] = dealii::DoFTools::none;
                else
                    coupling[c][d] = dealii::DoFTools::always;
            }


        dealii::BlockDynamicSparsityPattern dsp(relevant_partitioning);
        dealii::DoFTools::make_sparsity_pattern(dof_handler, 
                                                coupling, 
                                                dsp, 
                                                constraints, 
                                                /*keep_constrained_dofs*/false);
        dealii::SparsityTools::distribute_sparsity_pattern(
            dsp,
            dof_handler.locally_owned_dofs(),
            mpi_communicator,
            locally_relevant_dofs);

        system_matrix.reinit(owned_partitioning, dsp, mpi_communicator);
    }
    {
        dealii::Table<2, dealii::DoFTools::Coupling> preconditioner_coupling(3, 3);
        for (unsigned int c = 0; c < 3; ++c)
            for (unsigned int d = 0; d < 3; ++d)
            {
                if ( (c == 0) && (d == 0) )
                    preconditioner_coupling[c][d] = dealii::DoFTools::always;
                else
                    preconditioner_coupling[c][d] = dealii::DoFTools::none;
            }


        dealii::BlockDynamicSparsityPattern preconditioner_dsp(relevant_partitioning);
        dealii::DoFTools::make_sparsity_pattern(dof_handler, 
                                                preconditioner_coupling, 
                                                preconditioner_dsp, 
                                                constraints, 
                                                /*keep_constrained_dofs*/false);
        dealii::SparsityTools::distribute_sparsity_pattern(
            preconditioner_dsp,
            dof_handler.locally_owned_dofs(),
            mpi_communicator,
            locally_relevant_dofs);

        M_psi_matrix.reinit(owned_partitioning, preconditioner_dsp, mpi_communicator);
    }

    system_rhs.reinit(owned_partitioning, mpi_communicator);
}



template <int dim>
void PhaseFieldCrystalSystemMPI<dim>::initialize_fe_field()
{

    dealii::LinearAlgebraTrilinos::MPI::BlockVector Psi_0(owned_partitioning, 
                                                          mpi_communicator);
    dealii::VectorTools::interpolate(dof_handler,
                                     *initial_condition,
                                     Psi_0);
    constraints.distribute(Psi_0);
    Psi_0.compress(dealii::VectorOperation::insert);

    Psi_n.reinit(owned_partitioning, relevant_partitioning, mpi_communicator);
    Psi_n_1.reinit(owned_partitioning, relevant_partitioning, mpi_communicator);

    Psi_n = Psi_0;
    Psi_n_1 = Psi_0;
}



template <int dim>
void PhaseFieldCrystalSystemMPI<dim>::assemble_system()
{
    system_matrix = 0;
    system_rhs = 0;

    dealii::QGauss<dim> quadrature_formula(fe_system.degree + 1);
    
    dealii::FEValues<dim> fe_values(fe_system,
                                    quadrature_formula,
                                    dealii::update_values |
                                    dealii::update_gradients |
                                    dealii::update_quadrature_points |
                                    dealii::update_JxW_values);

    const unsigned int dofs_per_cell = fe_system.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();

    dealii::FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    dealii::FullMatrix<double> local_preconditioner_matrix(dofs_per_cell, dofs_per_cell);
    dealii::Vector<double> local_rhs(dofs_per_cell);

    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    const dealii::FEValuesExtractors::Scalar psi(0);
    const dealii::FEValuesExtractors::Scalar chi(1);
    const dealii::FEValuesExtractors::Scalar phi(2);

    std::vector<double> eta_psi(dofs_per_cell);
    std::vector<double> eta_chi(dofs_per_cell);
    std::vector<double> eta_phi(dofs_per_cell);
    std::vector<dealii::Tensor<1, dim>> grad_eta_psi(dofs_per_cell);
    std::vector<dealii::Tensor<1, dim>> grad_eta_chi(dofs_per_cell);
    std::vector<dealii::Tensor<1, dim>> grad_eta_phi(dofs_per_cell);

    std::vector<double> psi_n(n_q_points);
    std::vector<double> chi_n(n_q_points);
    std::vector<double> phi_n(n_q_points);
    std::vector<double> psi_n_1(n_q_points);
    std::vector<double> chi_n_1(n_q_points);
    std::vector<double> phi_n_1(n_q_points);
    std::vector<dealii::Tensor<1, dim>> grad_psi_n(n_q_points);
    std::vector<dealii::Tensor<1, dim>> grad_chi_n(n_q_points);
    std::vector<dealii::Tensor<1, dim>> grad_phi_n(n_q_points);
    std::vector<dealii::Tensor<1, dim>> grad_psi_n_1(n_q_points);
    std::vector<dealii::Tensor<1, dim>> grad_chi_n_1(n_q_points);
    std::vector<dealii::Tensor<1, dim>> grad_phi_n_1(n_q_points);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (!(cell->is_locally_owned()))
            continue;

        fe_values.reinit(cell);
        local_matrix = 0;
        local_preconditioner_matrix = 0;
        local_rhs = 0;

        for (const unsigned int q : fe_values.quadrature_point_indices())
        {
            fe_values[psi].get_function_values(Psi_n, psi_n);
            fe_values[chi].get_function_values(Psi_n, chi_n);
            fe_values[phi].get_function_values(Psi_n, phi_n);
            fe_values[psi].get_function_values(Psi_n_1, psi_n_1);
            fe_values[chi].get_function_values(Psi_n_1, chi_n_1);
            fe_values[phi].get_function_values(Psi_n_1, phi_n_1);
            fe_values[psi].get_function_gradients(Psi_n, grad_psi_n);
            fe_values[chi].get_function_gradients(Psi_n, grad_chi_n);
            fe_values[phi].get_function_gradients(Psi_n, grad_phi_n);
            fe_values[psi].get_function_gradients(Psi_n_1, grad_psi_n_1);
            fe_values[chi].get_function_gradients(Psi_n_1, grad_chi_n_1);
            fe_values[phi].get_function_gradients(Psi_n_1, grad_phi_n_1);

            for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
                grad_eta_psi[k] = fe_values[psi].gradient(k, q);
                grad_eta_chi[k] = fe_values[chi].gradient(k, q);
                grad_eta_phi[k] = fe_values[phi].gradient(k, q);
                eta_psi[k] = fe_values[psi].value(k, q);
                eta_chi[k] = fe_values[chi].value(k, q);
                eta_phi[k] = fe_values[phi].value(k, q);
            }

            for (const unsigned int i : fe_values.dof_indices())
            {
                local_rhs(i) +=
                    (
                     (eta_psi[i]
                      * (-psi_n[q] + psi_n_1[q]
                         + dt * theta * (2 * phi_n[q]
                                         + (1 + eps) * chi_n[q]
                                         + 3 * psi_n[q] * psi_n[q] * chi_n[q]
                                         + 6 * psi_n[q] * grad_psi_n[q] * grad_psi_n[q]
                             )
                         + dt * (1 - theta) * (2 * phi_n_1[q]
                                               + (1 + eps) * chi_n_1[q]
                                               + 3 * psi_n_1[q] * psi_n_1[q] * chi_n_1[q]
                                               + 6 * psi_n_1[q] * grad_psi_n_1[q] * grad_psi_n_1[q]
                             )
                      )
                     )
                     -
                     (dt * grad_eta_psi[i]
                      * (theta * grad_phi_n[q] + (1 - theta) * grad_phi_n_1[q])
                     )
                     -
                     (eta_chi[i] * chi_n[q])
                     -
                     (grad_eta_chi[i] * grad_psi_n[q])
                     -
                     (eta_phi[i] * phi_n[q])
                     -
                     (grad_eta_phi[i] * grad_chi_n[q])
                    )
                    * fe_values.JxW(q);

                for (const unsigned int j : fe_values.dof_indices())
                {
                    local_matrix(i, j) +=
                        (
                         (eta_psi[i] 
                          * (eta_psi[j] 
                             - dt * theta * (2 * eta_phi[j]
                                             + (1 + eps) * eta_chi[j]
                                             + 3 * psi_n[q] * psi_n[q] * eta_chi[j]
                                             + 6 * psi_n[q] * chi_n[q] * eta_psi[j]
                                             + 6 * grad_psi_n[q] * grad_psi_n[q] * eta_psi[j]
                                             + 12 * psi_n[q] * grad_psi_n[q] * grad_eta_psi[j])
                            )
                         )
                         +
                         (dt * theta * grad_eta_psi[i] * grad_eta_phi[j])
                         +
                         (eta_chi[i] * eta_chi[j])
                         +
                         (grad_eta_chi[i] * grad_eta_psi[j])
                         +
                         (eta_phi[i] * eta_phi[j])
                         +
                         (grad_eta_phi[i] * grad_eta_chi[j])
                        )
                        * fe_values.JxW(q); 

                    local_preconditioner_matrix(i, j) +=
                        (eta_psi[i] * eta_psi[j])
                        * fe_values.JxW(q); 
                }
            }
        }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(local_matrix,
                                               local_rhs,
                                               local_dof_indices,
                                               system_matrix,
                                               system_rhs);
        constraints.distribute_local_to_global(local_preconditioner_matrix,
                                               local_dof_indices,
                                               M_psi_matrix);
    }

    system_matrix.compress(dealii::VectorOperation::add);
    system_rhs.compress(dealii::VectorOperation::add);
    M_psi_matrix.compress(dealii::VectorOperation::add);
}



template <int dim>
void PhaseFieldCrystalSystemMPI<dim>::solve_and_update()
{
    using vec = dealii::LinearAlgebraTrilinos::MPI::Vector;
    using block_vec = dealii::LinearAlgebraTrilinos::MPI::BlockVector;

    const auto op_M_chi = dealii::linear_operator<vec>(system_matrix.block(1, 1));
    dealii::LinearAlgebraTrilinos::MPI::PreconditionAMG precondition_M_chi;
    precondition_M_chi.initialize(system_matrix.block(1, 1));

    dealii::ReductionControl reduction_control_M_chi(2000, 1e-18, 1e-10);
    dealii::SolverCG<dealii::LinearAlgebraTrilinos::MPI::Vector> 
        solver_cg_chi(reduction_control_M_chi);
    const auto M_chi_inv = dealii::inverse_operator(op_M_chi, 
                                                    solver_cg_chi, 
                                                    precondition_M_chi);

    const auto op_M_phi = dealii::linear_operator<vec>(system_matrix.block(2, 2));
    dealii::LinearAlgebraTrilinos::MPI::PreconditionAMG precondition_M_phi;
    precondition_M_phi.initialize(system_matrix.block(2, 2));

    dealii::ReductionControl reduction_control_M_phi(2000, 1e-18, 1e-10);
    dealii::SolverCG<dealii::LinearAlgebraTrilinos::MPI::Vector> 
        solver_cg_phi(reduction_control_M_phi);
    const auto M_phi_inv = dealii::inverse_operator(op_M_phi, 
                                                    solver_cg_phi, 
                                                    precondition_M_phi);

    const auto L_psi = dealii::linear_operator<vec>(system_matrix.block(1, 0));
    const auto L_chi = dealii::linear_operator<vec>(system_matrix.block(2, 1));

    const auto op_B = dealii::linear_operator<vec>(system_matrix.block(0, 0));
    dealii::LinearAlgebraTrilinos::MPI::PreconditionAMG precondition_B;
    precondition_B.initialize(system_matrix.block(0, 0));

    dealii::ReductionControl reduction_control_B(2000, 1e-18, 1e-10);
    dealii::SolverGMRES<dealii::LinearAlgebraTrilinos::MPI::Vector> 
        solver_gmres_B(reduction_control_B);
    const auto B_inv = dealii::inverse_operator(op_B, 
                                                solver_gmres_B, 
                                                precondition_B);

    // dealii::LinearAlgebraTrilinos::MPI::PreconditionAMG B_preconditioner;
    // B_preconditioner.initialize(system_matrix.block(0, 0));
    // const auto B_inv = dealii::linear_operator<vec>(B_preconditioner);

    const auto zero = dealii::null_operator(L_psi);

    const auto P_inv = dealii::block_operator<3, 3, block_vec>(
                       {B_inv, zero, zero,
                        -1.0 * M_chi_inv*L_psi*B_inv, M_chi_inv, zero,
                        M_phi_inv*L_chi*M_chi_inv*L_psi*B_inv, -1.0 * M_phi_inv*L_chi*M_chi_inv, M_phi_inv});

    dealii::SolverControl solver_control(dof_handler.n_dofs());

    dealii::SolverGMRES<block_vec> solver_gmres(solver_control);
    dealii::LinearAlgebraTrilinos::MPI::BlockVector dPsi_n(owned_partitioning, mpi_communicator);
    // solver_gmres.solve(system_matrix, dPsi_n, system_rhs, P_inv);
    solver_gmres.solve(system_matrix, dPsi_n, system_rhs, dealii::PreconditionIdentity());
    pcout << "Number of iterations: " << solver_control.last_step() << "\n";

    constraints.distribute(dPsi_n);

    dealii::LinearAlgebraTrilinos::MPI::BlockVector 
        local_Psi_n(owned_partitioning, mpi_communicator);
    local_Psi_n = Psi_n;
    local_Psi_n += dPsi_n;
    Psi_n = local_Psi_n;
}



template <int dim>
void PhaseFieldCrystalSystemMPI<dim>::iterate_timestep()
{
    for (unsigned int n_iters = 0; n_iters < simulation_max_iters; ++n_iters)
    {
        pcout << "Assembling system!\n";
        assemble_system();

        const double residual = system_rhs.l2_norm();
        pcout << "Residual is: " << residual << "\n";
        if (residual < simulation_tol)
            break;

        pcout << "Solving and updating!\n";
        solve_and_update();
    }
    Psi_n_1 = Psi_n;
}



template <int dim>
void PhaseFieldCrystalSystemMPI<dim>::output_configuration(unsigned int iteration)
{
    std::vector<std::string> field_names = {"psi", "chi", "phi"};
    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(3, dealii::DataComponentInterpretation::component_is_scalar);

    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(Psi_n,
                             field_names,
                             dealii::DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    data_out.build_patches();

    data_out.write_vtu_with_pvtu_record("./", "phase_field_", iteration,
                                        mpi_communicator,
                                        /*n_digits_for_counter*/2);
}



template <int dim>
void PhaseFieldCrystalSystemMPI<dim>::output_rhs(unsigned int iteration)
{
    std::vector<std::string> field_names = {"psi", "chi", "phi"};
    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(3, dealii::DataComponentInterpretation::component_is_scalar);

    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(system_rhs,
                             field_names,
                             dealii::DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    data_out.build_patches();

    data_out.write_vtu_with_pvtu_record("./", "phase_field_rhs_", iteration,
                                        mpi_communicator,
                                        /*n_digits_for_counter*/2);
}



template <int dim>
void PhaseFieldCrystalSystemMPI<dim>::run()
{
    pcout << "Making grid!\n";
    make_grid();

    pcout << "Setting up dofs!\n";
    setup_dofs();

    pcout << "Initializing fe field!\n";
    initialize_fe_field();

    const unsigned int n_timesteps = 10000;
    for (unsigned int timestep = 0; timestep < n_timesteps; ++timestep)
    {
        pcout << "Outputting configuration!\n";
        output_configuration(timestep);

        pcout << "Outputting right-hand side!\n";
        output_rhs(timestep);


        pcout << "\n";
        pcout << "Iterating timestep: " << timestep << "\n";
        iterate_timestep();
    }
}


template class PhaseFieldCrystalSystemMPI<2>;
template class PhaseFieldCrystalSystemMPI<3>;
