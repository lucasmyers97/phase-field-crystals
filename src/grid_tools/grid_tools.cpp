#include "grid_tools.hpp"

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>

namespace grid_tools {

template <int dim>
using cell_iterator = typename dealii::Triangulation<dim>::active_cell_iterator;

template <int dim>
void visit_neighborhood(dealii::Triangulation<dim> & tria,
                        const cell_iterator<dim>& base_cell, 
                        std::function<bool(const cell_iterator<dim>&)> &is_in_neighborhood,
                        std::function<void(const cell_iterator<dim>&)> &calculate_local_quantity,
                        bool clear_user_flags)
{
    if (clear_user_flags)
        tria.clear_user_flags();

    visit_neighbors_recursively<dim>(base_cell, 
                                     is_in_neighborhood, 
                                     calculate_local_quantity);

    if (clear_user_flags)
        tria.clear_user_flags();
}



template <int dim>
void visit_neighbors_recursively(const cell_iterator<dim>& cell, 
                                 std::function<bool(const cell_iterator<dim>&)> &is_in_neighborhood,
                                 std::function<void(const cell_iterator<dim>&)> &calculate_local_quantity)
{
    calculate_local_quantity(cell);
    cell->set_user_flag();

    for (unsigned int i = 0; i < cell->n_faces(); ++i)
    {
        if (cell->at_boundary(i))
            continue;
 
        const auto neighbor = cell->neighbor(i);
        if (neighbor->has_children())
            for (unsigned int j = 0; j < neighbor->n_children(); ++j)
            {
                const auto child = neighbor->child(j);
                if (child->is_artificial() 
                    || child->user_flag_set() 
                    || !is_in_neighborhood(child))
                    continue;
 
                visit_neighbors_recursively<dim>(child, is_in_neighborhood, calculate_local_quantity);
            }
        else
        {
            if (neighbor->is_artificial() 
                || neighbor->user_flag_set() 
                || !is_in_neighborhood(neighbor))
                continue;
 
            visit_neighbors_recursively<dim>(neighbor, is_in_neighborhood, calculate_local_quantity);
        }
    }
}



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
bool IsInL2Neighborhood<dim>::operator()(const cell_iterator<dim> &cell)
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
void visit_neighborhood<2>(dealii::Triangulation<2> &tria, const cell_iterator<2> &base_cell, 
                           std::function<bool (const cell_iterator<2> &)> &is_in_neighborhood, 
                           std::function<void (const cell_iterator<2> &)> &calculate_local_quantity, 
                           bool clear_user_flags);

template
void visit_neighborhood<3>(dealii::Triangulation<3> &tria, const cell_iterator<3> &base_cell, 
                           std::function<bool (const cell_iterator<3> &)> &is_in_neighborhood, 
                           std::function<void (const cell_iterator<3> &)> &calculate_local_quantity, 
                           bool clear_user_flags);

template
class neighborhood_functions::IsInL2Neighborhood<2>;

template
class neighborhood_functions::IsInL2Neighborhood<3>;

} // grid_tools
