#include "phase_field_crystal_system.hpp"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/types.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <vector>

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
    dealii::GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(n_refines);
}



template <int dim>
void PhaseFieldCrystalSystem<dim>::setup_dofs()
{
    dof_handler.distribute_dofs(fe_system);

    std::vector<unsigned int> block_component = {0, 1, 2};
    const std::vector<dealii::types::global_dof_index> 
        dofs_per_block = dealii::DoFTools::count_dofs_per_fe_block(dof_handler, block_component);

    psi_n.reinit(dofs_per_block);
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
void PhaseFieldCrystalSystem<dim>::run(unsigned int n_refines)
{
    make_grid(n_refines);
    setup_dofs();
}


template class PhaseFieldCrystalSystem<2>;
template class PhaseFieldCrystalSystem<3>;
