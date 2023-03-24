#include "phase_field_crystal_system.hpp"

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <fstream>
#include <iostream>

template <int dim>
PhaseFieldCrystalSystem<dim>::PhaseFieldCrystalSystem()
{}



template <int dim>
void PhaseFieldCrystalSystem<dim>::run(unsigned int n_refines)
{
    dealii::GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(n_refines);

    std::ofstream output_stream("grid.svg");
    dealii::GridOut grid_out;
    grid_out.write_svg(triangulation, output_stream);

    std::cout << "Outputted grid\n";
}


template class PhaseFieldCrystalSystem<2>;
template class PhaseFieldCrystalSystem<3>;
