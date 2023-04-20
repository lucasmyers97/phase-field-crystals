#include "phase_field_crystal_system.hpp"

#include <deal.II/base/mpi.h>

#include <deal.II/base/array_view.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/types.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/grid/tria.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/numerics/data_component_interpretation.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>

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
PhaseFieldCrystalSystem<dim>::PhaseFieldCrystalSystem(unsigned int degree)
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
{}



template <int dim>
void PhaseFieldCrystalSystem<dim>::make_grid(unsigned int n_refines)
{
    const double a = 4 * M_PI / std::sqrt(3);

    // const int n_across = 5;
    // const int n_down = 5;
    // const double left = -n_across * (2 / std::sqrt(3)) * M_PI * 2;
    // const double down = -n_down * 2 * M_PI;
    // const double left = -20 * a;
    // const double down = -20 * std::sqrt(3) * a;
    const double left = -6 * a;
    const double down = -6 * a;

    dealii::Point<dim> p1 = {left, down};
    dealii::Point<dim> p2 = -p1;

    dealii::GridGenerator::hyper_rectangle(triangulation, 
                                           p1, 
                                           p2, 
                                           /*colorize*/ true);
    triangulation.refine_global(n_refines);
}



template <int dim>
void PhaseFieldCrystalSystem<dim>::setup_dofs()
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
    relevant_partitioning[0] = dof_handler
                               .locally_owned_dofs()
                               .get_view(0, n_psi);
    relevant_partitioning[1] = dof_handler
                               .locally_owned_dofs()
                               .get_view(n_psi, n_psi + n_chi);
    relevant_partitioning[2] = dof_handler
                               .locally_owned_dofs()
                               .get_view(n_psi + n_chi, n_psi + n_chi + n_phi);
    {
        constraints.clear();
        constraints.reinit(locally_relevant_dofs);
        dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);
        dealii::DoFTools::make_periodicity_constraints(dof_handler,
                                                       /*b_id1*/ 0,
                                                       /*b_id2*/ 1,
                                                       /*direction*/ 0,
                                                       constraints);
        dealii::DoFTools::make_periodicity_constraints(dof_handler,
                                                       /*b_id1*/ 2,
                                                       /*b_id2*/ 3,
                                                       /*direction*/ 1,
                                                       constraints);
        constraints.close();
    }
    {
        dealii::Table<2, dealii::DoFTools::Coupling> coupling(3, 3);
        for (unsigned int c = 0; c < 3; ++c)
            for (unsigned int d = 0; d < 3; ++d)
                if ( ((c == 2) && (d == 0))
                    ||((c == 1) && (d == 2)) )
                    coupling[c][d] = dealii::DoFTools::none;
                else
                    coupling[c][d] = dealii::DoFTools::always;

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

    system_rhs.reinit(owned_partitioning, mpi_communicator);
    dPsi_n.reinit(owned_partitioning, relevant_partitioning, mpi_communicator);
}



template <int dim>
void PhaseFieldCrystalSystem<dim>::initialize_fe_field()
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

    HexagonalLattice<dim> hexagonal_lattice(A_0, 
                                            psi_0, 
                                            dislocation_positions, 
                                            burgers_vectors);

    dealii::LinearAlgebraTrilinos::MPI::BlockVector Psi_0(owned_partitioning, 
                                                          mpi_communicator);

    dealii::VectorTools::project(dof_handler,
                                 constraints,
                                 dealii::QGauss<dim>(fe_system.degree + 1),
                                 hexagonal_lattice,
                                 Psi_0);

    Psi_n.reinit(owned_partitioning, relevant_partitioning, mpi_communicator);
    Psi_n_1.reinit(owned_partitioning, relevant_partitioning, mpi_communicator);

    Psi_n = Psi_0;
    Psi_n_1 = Psi_0;
}



template <int dim>
void PhaseFieldCrystalSystem<dim>::assemble_system()
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
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);
        local_matrix = 0;
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
                }
            }
        }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(local_matrix,
                                               local_rhs,
                                               local_dof_indices,
                                               system_matrix,
                                               system_rhs);
    }

    system_matrix.compress(dealii::VectorOperation::add);
    system_rhs.compress(dealii::VectorOperation::add);
}



template <int dim>
void PhaseFieldCrystalSystem<dim>::solve_and_update()
{
    dealii::SparseDirectUMFPACK A_inv;
    A_inv.factorize(system_matrix);
    A_inv.vmult(dPsi_n, system_rhs);

    // dealii::FullMatrix<double> psi_matrix;

    // dealii::FullMatrix<double> M_chi_inv;
    // M_chi_inv.copy_from(system_matrix.block(1, 1));
    // M_chi_inv.invert(M_chi_inv);

    // dealii::FullMatrix<double> M_phi_inv;
    // M_phi_inv.copy_from(system_matrix.block(2, 2));
    // M_phi_inv.invert(M_phi_inv);
    // {
    //     dealii::FullMatrix<double> E(system_rhs.block(1).size(),
    //                                  system_rhs.block(0).size());
    //     {
    //         dealii::FullMatrix<double> L_psi;
    //         L_psi.copy_from(system_matrix.block(1, 0));

    //         M_chi_inv.mmult(E, L_psi);
    //     }

    //     dealii::FullMatrix<double> F(system_rhs.block(2).size(),
    //                                  system_rhs.block(1).size());
    //     {
    //         dealii::FullMatrix<double> L_chi;
    //         L_chi.copy_from(system_matrix.block(2, 1));
    //         M_phi_inv.mmult(F, L_chi);
    //     }

    //     dealii::FullMatrix<double> C;
    //     C.copy_from(system_matrix.block(0, 1));
    //     C *= -1;

    //     {
    //         dealii::FullMatrix<double> D;
    //         D.copy_from(system_matrix.block(0, 2));

    //         D.mmult(C, F, true); 
    //     }

    //     C.mmult(C, E);

    //     psi_matrix.copy_from(system_matrix.block(0, 0));
    //     psi_matrix.add(1.0, C);
    // }
    // psi_matrix.invert(psi_matrix);

    // dealii::Vector<double> psi_rhs(system_rhs.block(0).size());
    // {
    //     dealii::Vector<double> tmp_psi(system_rhs.block(0).size());
    //     dealii::Vector<double> tmp_chi(system_rhs.block(1).size());
    //     dealii::Vector<double> tmp_phi_1(system_rhs.block(2).size());
    //     dealii::Vector<double> tmp_phi_2(system_rhs.block(2).size());

    //     M_chi_inv.vmult(tmp_chi, system_rhs.block(1));

    //     system_matrix.block(2, 1).vmult(tmp_phi_1, tmp_chi);
    //     tmp_phi_1 -= system_rhs.block(2);
    //     M_phi_inv.vmult(tmp_phi_2, tmp_phi_1);
    //     system_matrix.block(0, 2).vmult(psi_rhs, tmp_phi_2);

    //     system_matrix.block(0, 1).vmult(tmp_psi, tmp_chi);

    //     psi_rhs -= tmp_psi;
    //     psi_rhs += system_rhs.block(0);
    // }

    // // dealii::SolverControl psi_solver_control(1000);
    // // dealii::SolverGMRES<dealii::Vector<double>> psi_solver(psi_solver_control);
    // // psi_solver.solve<PsiMatrix, dealii::PreconditionIdentity>
    // //                 (psi_matrix, 
    // //                  dPsi_n.block(0), 
    // //                  psi_rhs, 
    // //                  dealii::PreconditionIdentity());

    // psi_matrix.vmult(dPsi_n.block(0), psi_rhs);

    // dealii::Vector<double> chi_rhs(system_rhs.block(1));
    // chi_rhs *= -1;
    // system_matrix.block(1, 0).vmult_add(chi_rhs, dPsi_n.block(0));
    // chi_rhs *= -1;
    // M_chi_inv.vmult(dPsi_n.block(1), chi_rhs);

    // dealii::Vector<double> phi_rhs(system_rhs.block(2));
    // phi_rhs *= -1;
    // system_matrix.block(2, 1).vmult_add(phi_rhs, dPsi_n.block(1));
    // phi_rhs *= -1;
    // M_phi_inv.vmult(dPsi_n.block(2), phi_rhs);
    
    constraints.distribute(dPsi_n);
    Psi_n += dPsi_n;
}



template <int dim>
void PhaseFieldCrystalSystem<dim>::iterate_timestep()
{
    for (unsigned int n_iters = 0; n_iters < simulation_max_iters; ++n_iters)
    {
        std::cout << "Assembling system!\n";
        assemble_system();

        const double residual = system_rhs.l2_norm();
        std::cout << "Residual is: " << residual << "\n";
        if (residual < simulation_tol)
            break;

        std::cout << "Solving and updating!\n";
        solve_and_update();
    }
    Psi_n_1 = Psi_n;
}



template <int dim>
void PhaseFieldCrystalSystem<dim>::output_configuration(unsigned int iteration)
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

    std::ofstream output(std::string("phase_field_") 
                         + std::to_string(iteration)
                         + std::string(".vtu"));
    data_out.write_vtu(output);
}



template <int dim>
void PhaseFieldCrystalSystem<dim>::output_rhs(unsigned int iteration)
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

    std::ofstream output(std::string("phase_field_rhs_") 
                         + std::to_string(iteration)
                         + std::string(".vtu"));
    data_out.write_vtu(output);
}



template <int dim>
void PhaseFieldCrystalSystem<dim>::run(unsigned int n_refines)
{
    std::cout << "Making grid!\n";
    make_grid(n_refines);

    std::cout << "Setting up dofs!\n";
    setup_dofs();

    std::cout << "Initializing fe field!\n";
    initialize_fe_field();

    const unsigned int n_timesteps = 10000;
    for (unsigned int timestep = 0; timestep < n_timesteps; ++timestep)
    {
        std::cout << "Outputting configuration!\n";
        output_configuration(timestep);

        std::cout << "Outputting right-hand side!\n";
        output_rhs(timestep);


        std::cout << "Iterating timestep: " << timestep << "\n";
        iterate_timestep();
        std::cout << "\n";
    }
}


template class PhaseFieldCrystalSystem<2>;
template class PhaseFieldCrystalSystem<3>;
