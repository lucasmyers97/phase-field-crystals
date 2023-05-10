#include "stress_calculator_mpi.hpp"

#include <deal.II/base/types.h>
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



template class StressCalculatorMPI<2>;
template class StressCalculatorMPI<3>;
