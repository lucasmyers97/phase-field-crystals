#include "phase_field_crystal_system.hpp"

#include <deal.II/base/array_view.h>
#include <deal.II/base/quadrature_lib.h>
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
#include <deal.II/numerics/data_component_interpretation.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>
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
    dealii::GridGenerator::hyper_cube(triangulation, 
                                      /*left*/ -20.0, 
                                      /*right*/ 20.0, 
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
                    ||((c == 1) && (d == 3)) )
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
    dpsi_n.reinit(dofs_per_block);
    psi_n.reinit(dofs_per_block);
    psi_n_1.reinit(dofs_per_block);
}



template <int dim>
void PhaseFieldCrystalSystem<dim>::initialize_fe_field()
{
    HexagonalLattice<dim> hexagonal_lattice;

    dealii::VectorTools::project(dof_handler,
                                 constraints,
                                 dealii::QGauss<dim>(fe_system.degree + 1),
                                 hexagonal_lattice,
                                 psi_n);
}



template <int dim>
void PhaseFieldCrystalSystem<dim>::output_configuration()
{
    std::vector<std::string> field_names = {"psi", "chi", "phi"};
    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(3, dealii::DataComponentInterpretation::component_is_scalar);

    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(psi_n,
                             field_names,
                             dealii::DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    data_out.build_patches();

    std::ofstream output("phase_field.vtu");
    data_out.write_vtu(output);
}



template <int dim>
void PhaseFieldCrystalSystem<dim>::run(unsigned int n_refines)
{
    make_grid(n_refines);
    setup_dofs();
    initialize_fe_field();
    output_configuration();
}


template class PhaseFieldCrystalSystem<2>;
template class PhaseFieldCrystalSystem<3>;
