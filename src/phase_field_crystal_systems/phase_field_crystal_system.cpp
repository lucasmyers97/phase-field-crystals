#include "phase_field_crystal_system.hpp"

#include <cstdlib>
#include <deal.II/base/array_view.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/base/tensor.h>
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
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/numerics/data_component_interpretation.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/linear_operator_tools.h>

#include <fstream>
#include <iostream>
#include <limits>
#include <vector>
#include <string>

#include "phase_field_functions/hexagonal_lattice.hpp"



template <int dim>
PhaseFieldCrystalSystem<dim>::PhaseFieldCrystalSystem(unsigned int degree)
    : fe_system(dealii::FE_Q<dim>(degree), 1,
                dealii::FE_Q<dim>(degree), 1,
                dealii::FE_Q<dim>(degree), 1)
    , dof_handler(triangulation)
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
    const double left = -2 * a;
    const double down = -2 * a;

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
    system_matrix.clear();

    dof_handler.distribute_dofs(fe_system);

    std::vector<unsigned int> block_component = {0, 1, 2};
    dealii::DoFRenumbering::component_wise(dof_handler, block_component);

    constraints.clear();
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

    const std::vector<dealii::types::global_dof_index> 
        dofs_per_block = dealii::DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
    {
        dealii::BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);
        dealii::Table<2, dealii::DoFTools::Coupling> coupling(3, 3);
        for (unsigned int c = 0; c < 3; ++c)
            for (unsigned int d = 0; d < 3; ++d)
                if ( ((c == 2) && (d == 0))
                    ||((c == 1) && (d == 2)) )
                    coupling[c][d] = dealii::DoFTools::none;
                else
                    coupling[c][d] = dealii::DoFTools::always;

        dealii::DoFTools::make_sparsity_pattern(dof_handler, 
                                                coupling, 
                                                dsp, 
                                                constraints, 
                                                /*keep_constrained_dofs*/false);
        sparsity_pattern.copy_from(dsp);
    }
    system_matrix.reinit(sparsity_pattern);

    system_rhs.reinit(dofs_per_block);
    dPsi_n.reinit(dofs_per_block);
    Psi_n.reinit(dofs_per_block);
    Psi_n_1.reinit(dofs_per_block);
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

    dealii::VectorTools::project(dof_handler,
                                 constraints,
                                 dealii::QGauss<dim>(fe_system.degree + 1),
                                 hexagonal_lattice,
                                 Psi_n);
    Psi_n_1 = Psi_n;
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
}



template <int dim>
void PhaseFieldCrystalSystem<dim>::solve_and_update()
{
    const auto op_M_chi = dealii::linear_operator(system_matrix.block(1, 1));
    dealii::PreconditionJacobi<dealii::SparseMatrix<double>> precondition_M_chi;
    precondition_M_chi.initialize(system_matrix.block(1, 1));

    dealii::ReductionControl reduction_control_M_chi(2000, 1e-18, 1e-10);
    dealii::SolverCG<dealii::Vector<double>> solver_cg_chi(reduction_control_M_chi);
    const auto M_chi_inv = dealii::inverse_operator(op_M_chi, 
                                                    solver_cg_chi, 
                                                    precondition_M_chi);

    const auto op_M_phi = dealii::linear_operator(system_matrix.block(2, 2));
    dealii::PreconditionJacobi<dealii::SparseMatrix<double>> precondition_M_phi;
    precondition_M_phi.initialize(system_matrix.block(2, 2));

    dealii::ReductionControl reduction_control_M_phi(2000, 1e-18, 1e-10);
    dealii::SolverCG<dealii::Vector<double>> solver_cg_phi(reduction_control_M_phi);
    const auto M_phi_inv = dealii::inverse_operator(op_M_phi, 
                                                    solver_cg_phi, 
                                                    precondition_M_phi);
    const auto B = dealii::linear_operator(system_matrix.block(0, 0));
    const auto C = dealii::linear_operator(system_matrix.block(0, 1));
    const auto D = dealii::linear_operator(system_matrix.block(0, 2));
    const auto L_psi = dealii::linear_operator(system_matrix.block(1, 0));
    const auto L_chi = dealii::linear_operator(system_matrix.block(2, 1));

    const auto psi_matrix = B + (D*M_phi_inv*L_chi - C) * M_chi_inv*L_psi;

    const dealii::Vector<double>& F = system_rhs.block(0);
    const dealii::Vector<double>& G = system_rhs.block(1);
    const dealii::Vector<double>& H = system_rhs.block(2);

    const auto psi_rhs = F 
                         - C * M_chi_inv * G 
                         + D * M_phi_inv * (L_chi * M_chi_inv * G - H);

    dealii::SolverControl solver_control(500);
    dealii::SolverGMRES<dealii::Vector<double>> solver_gmres(solver_control);
    solver_gmres.solve(psi_matrix, dPsi_n.block(0), psi_rhs, dealii::PreconditionIdentity());

    const auto chi_rhs = M_chi_inv * (G - L_psi * dPsi_n.block(0));
    chi_rhs.apply(dPsi_n.block(1));

    const auto phi_rhs = M_phi_inv * (H - L_chi * dPsi_n.block(1));
    phi_rhs.apply(dPsi_n.block(2));
    
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

    const unsigned int n_timesteps = 100;
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
