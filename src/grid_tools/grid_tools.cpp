#include "grid_tools.hpp"

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>

namespace grid_tools {

template <int dim>
using tria_cell_iterator = typename dealii::Triangulation<dim>::active_cell_iterator;

namespace neighborhood_functions {

template <int dim>
IsInL2Neighborhood<dim>::IsInL2Neighborhood(dealii::FEValues<dim>& fe_values,
                                            const dealii::Point<dim>& base_point,
                                            double radius)
    : fe_values(fe_values)
    , base_point(base_point)
    , radius(radius)
{}



template<int dim>
bool IsInL2Neighborhood<dim>::operator()(const tria_cell_iterator<dim> &cell)
{
    fe_values.reinit(cell);
    const auto& quad_points = fe_values.get_quadrature_points();
    for (const auto& point : quad_points)
        if (base_point.distance(point) < radius)
            return true;
    
    return false;
}

}

template
class neighborhood_functions::IsInL2Neighborhood<2>;

template
class neighborhood_functions::IsInL2Neighborhood<3>;

} // grid_tools
