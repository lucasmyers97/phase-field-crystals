#include "stress_calculator_mpi.hpp"

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

template <int dim>
StressCalculatorMPI<dim>::
StressCalculatorMPI(const dealii::Triangulation<dim> &tria, 
                    const unsigned int degree)
    : dof_handler(tria)
    , fe_system(dealii::FE_Q<dim>(degree)^(dim*dim))
{}



template <int dim>
void StressCalculatorMPI<dim>::
setup_dofs(const MPI_Comm& mpi_communicator)
{
    dof_handler.distribute_dofs(fe_system);

    std::vector<unsigned int> block_component(dim * dim);
    for (std::size_t i = 0; i < dim*dim; ++i)
        block_component[i] = i;
    dealii::DoFRenumbering::component_wise(dof_handler, block_component);

    const std::vector<dealii::types::global_dof_index> 
        dofs_per_block = dealii::DoFTools::count_dofs_per_fe_block(dof_handler, block_component);

    owned_partitioning.resize(dim * dim);
    dealii::types::global_dof_index partition_start = 0;
    for (std::size_t i = 0; i < dim*dim; ++i)
    {
        owned_partitioning[i] = dof_handler
                                .locally_owned_dofs()
                                .get_view(partition_start, 
                                          partition_start + dofs_per_block[i]);
        partition_start += dofs_per_block[i];
    }

    const dealii::IndexSet locally_relevant_dofs =
        dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);
    relevant_partitioning.resize(dim * dim);
    partition_start = 0;
    for (std::size_t i = 0; i < dim*dim; ++i)
    {
        relevant_partitioning[i] = locally_relevant_dofs
                                   .get_view(partition_start, 
                                             partition_start + dofs_per_block[i]);
        partition_start += dofs_per_block[i];
    }

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
        dealii::Table<2, dealii::DoFTools::Coupling> coupling(dim*dim, dim*dim);
        for (unsigned int c = 0; c < dim*dim; ++c)
            for (unsigned int d = 0; d < dim*dim; ++d)
            {
                if (c == d)
                    coupling[c][d] = dealii::DoFTools::always;
                else
                    coupling[c][d] = dealii::DoFTools::none;
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

    system_rhs.reinit(owned_partitioning, mpi_communicator);
}



template <int dim>
void StressCalculatorMPI<dim>::calculate_mass_matrix()
{
    system_matrix = 0;

    dealii::QGauss<dim> quadrature_formula(fe_system.degree + 1);
    
    dealii::FEValues<dim> fe_values(fe_system,
                                    quadrature_formula,
                                    dealii::update_values |
                                    dealii::update_JxW_values);

    const unsigned int dofs_per_cell = fe_system.n_dofs_per_cell();

    dealii::FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);

    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<double> tau(dofs_per_cell);
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (!(cell->is_locally_owned()))
            continue;

        fe_values.reinit(cell);
        local_matrix = 0;

        for (const unsigned int q : fe_values.quadrature_point_indices())
        {
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
                tau[k] = fe_values.shape_value(k, q);

            for (const unsigned int i : fe_values.dof_indices())
            {
                const unsigned int component_i =
                    fe_system.system_to_component_index(i).first;

                for (const unsigned int j : fe_values.dof_indices())
                {
                    const unsigned int component_j =
                        fe_system.system_to_component_index(j).first;

                    local_matrix(i, j) +=
                        (component_i == component_j) ?
                        tau[i] * tau[j] * fe_values.JxW(q) :
                        0; 
                }
            }
        }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(local_matrix,
                                               local_dof_indices,
                                               system_matrix);
    }

    system_matrix.compress(dealii::VectorOperation::add);
}



template <int dim>
void StressCalculatorMPI<dim>::
calculate_righthand_side(const dealii::DoFHandler<dim> &Psi_dof_handler,
                         const dealii::LinearAlgebraTrilinos::MPI::BlockVector &Psi,
                         double eps)
{
    system_rhs = 0;

    dealii::QGauss<dim> quadrature_formula(fe_system.degree + 1);
    const unsigned int n_q_points = quadrature_formula.size();
    
    dealii::FEValues<dim> fe_values(fe_system,
                                    quadrature_formula,
                                    dealii::update_values |
                                    dealii::update_gradients |
                                    dealii::update_JxW_values);

    dealii::FEValues<dim> Psi_fe_values(Psi_dof_handler.get_fe(),
                                        quadrature_formula,
                                        dealii::update_values |
                                        dealii::update_gradients |
                                        dealii::update_JxW_values);

    const unsigned int dofs_per_cell = fe_system.n_dofs_per_cell();

    dealii::Vector<double> local_rhs(dofs_per_cell);

    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    const dealii::FEValuesExtractors::Tensor<2> tau_idx(0);

    const dealii::FEValuesExtractors::Scalar psi_idx(0);
    const dealii::FEValuesExtractors::Scalar chi_idx(1);
    const dealii::FEValuesExtractors::Scalar phi_idx(2);

    std::vector<double> psi(n_q_points);
    std::vector<double> chi(n_q_points);
    std::vector<double> phi(n_q_points);
    std::vector<dealii::Tensor<1, dim>> grad_psi(n_q_points);
    std::vector<dealii::Tensor<1, dim>> grad_chi(n_q_points);
    std::vector<dealii::Tensor<1, dim>> grad_phi(n_q_points);

    std::vector<dealii::Tensor<2, dim>> tau(dofs_per_cell);
    std::vector<dealii::Tensor<3, dim>> grad_tau(dofs_per_cell);
    std::vector<dealii::Tensor<1, dim>> div_tau(dofs_per_cell);
    std::vector<dealii::Tensor<1, dim>> grad_tr_tau(dofs_per_cell);

    auto cell = dof_handler.begin_active();
    const auto endc = dof_handler.end();
    auto Psi_cell = Psi_dof_handler.begin_active();

    for (; cell != endc; ++cell, ++Psi_cell)
    {
        if (!(cell->is_locally_owned()))
            continue;

        fe_values.reinit(cell);
        Psi_fe_values.reinit(Psi_cell);
        local_rhs = 0;

        Psi_fe_values[psi_idx].get_function_values(Psi, psi);
        Psi_fe_values[chi_idx].get_function_values(Psi, chi);
        Psi_fe_values[phi_idx].get_function_values(Psi, phi);
        Psi_fe_values[psi_idx].get_function_gradients(Psi, grad_psi);
        Psi_fe_values[chi_idx].get_function_gradients(Psi, grad_chi);
        Psi_fe_values[phi_idx].get_function_gradients(Psi, grad_phi);

        for (const unsigned int q : fe_values.quadrature_point_indices())
        {
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
                tau[k] = fe_values[tau_idx].value(k, q);
                div_tau[k] = fe_values[tau_idx].divergence(k, q);

                auto grad_tau = fe_values[tau_idx].gradient(k, q);
                grad_tr_tau[k] = 0;
                for (unsigned int i = 0; i < dim; ++i)
                    grad_tr_tau[k] += grad_tau[i][i];
            }

            for (const unsigned int i : fe_values.dof_indices())
                local_rhs(i) +=
                    (dealii::scalar_product(tau[i], 
                                            dealii::outer_product(grad_psi[q] + grad_chi[q], 
                                                                  grad_psi[q])
                                            +
                                            dealii::outer_product(grad_psi[q],
                                                                  grad_psi[q] + grad_chi[q]))
                     +
                     div_tau[i] * (psi[q] + chi[q]) * grad_psi[q]
                     +
                     0.5 * dealii::trace(tau[i]) * (chi[q]*chi[q] 
                                                    + (1 + eps) * psi[q]*psi[q] 
                                                    + 0.5 * psi[q]*psi[q]*psi[q]*psi[q])
                     -
                     grad_tr_tau[i] * psi[q] * grad_psi[q]
                     -
                     dealii::trace(tau[i]) * grad_psi[q] * grad_psi[q])
                    *
                    fe_values.JxW(q);
        }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(local_rhs,
                                               local_dof_indices,
                                               system_rhs);
    }

    system_rhs.compress(dealii::VectorOperation::add);
}



template class StressCalculatorMPI<2>;
template class StressCalculatorMPI<3>;
