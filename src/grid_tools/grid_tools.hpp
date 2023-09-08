#ifndef PHASE_FIELD_CRYSTALS_GRID_TOOLS_HPP
#define PHASE_FIELD_CRYSTALS_GRID_TOOLS_HPP

#include <deal.II/base/point.h>
#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_handler.h>

namespace grid_tools {
   
template <int dim>
using tria_cell_iterator = typename dealii::Triangulation<dim>::active_cell_iterator;

/**
 * \brief Visits neighborhood of `base_cell` and calculates some quantity using
 * each cell in the neighborhood.
 *
 * Here a "neighborhood" is a connected collection of cells around `base_cell`
 * for which `is_in_neighborhood` evaluates to `true` when the cells are 
 * inputted.
 * This is essentially a wrapper function around `visit_neighbors_recursively`,
 * but it clears user_flags before and after visiting the neighborhood, unless
 * users specifically request not.
 *
 * @param cell Cell which `calculate_local_quantity` is run on, and whose 
 * neighbors are visited recursively
 *
 * @param is_in_neighborhood Function which eats a cell and returns a boolean 
 * to tell whether it satisfies criteria for being in neighborhood.
 * Users may construct `is_in_neighborhood` to change its internal state upon 
 * evaluation.
 *
 * @param calculate_local_quantity Function which takes the current cell and 
 * calculates some quantity.
 * Users may construct `calculate_local_quantity` to change its internal state 
 * upon evaluation.
 */
template <int dim, typename F, typename G>
void visit_neighborhood(dealii::Triangulation<dim>& tria,
                        const tria_cell_iterator<dim>& base_cell, 
                        F &is_in_neighborhood,
                        G &calculate_local_quantity,
                        bool clear_user_flags=true);

/**
 * \brief Recursively visits each cell in neighborhood of a base cell and runs 
 * function on those cells.
 *
 * Given a cell `cell`, this function runs `calculate_local_quantity`, marks
 * `cell` user flag as having been visited, then recursively visits neighbors
 * if `is_in_neighborhood` evaluates to true on those neighbors. 
 *
 * @param cell Cell which `calculate_local_quantity` is run on, and whose 
 * neighbors are visited recursively
 *
 * @param is_in_neighborhood Function which eats a cell and returns a boolean 
 * to tell whether it satisfies criteria for being in neighborhood.
 * Users may construct `is_in_neighborhood` to change its internal state upon 
 * evaluation.
 *
 * @param calculate_local_quantity Function which takes the current cell and 
 * calculates some quantity.
 * Users may construct `calculate_local_quantity` to change its internal state 
 * upon evaluation.
 */
template <int dim, typename F, typename G>
void visit_neighbors_recursively(const tria_cell_iterator<dim>& cell, 
                                 F &is_in_neighborhood,
                                 G &calculate_local_quantity);

namespace neighborhood_functions {

template <int dim>
class IsInL2Neighborhood 
{
public:
    IsInL2Neighborhood(dealii::FEValues<dim>& fe_values,
                       const dealii::Point<dim>& base_point,
                       double radius);
    bool operator()(const tria_cell_iterator<dim>& cell);

private:
    dealii::FEValues<dim> &fe_values;
    const dealii::Point<dim> &base_point;
    const double radius;
};

}

template <int dim, typename F, typename G>
void visit_neighborhood(dealii::Triangulation<dim> & tria,
                        const tria_cell_iterator<dim>& base_cell, 
                        F &is_in_neighborhood,
                        G &calculate_local_quantity,
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



template <int dim, typename F, typename G>
void visit_neighbors_recursively(const tria_cell_iterator<dim>& cell, 
                                 F &is_in_neighborhood,
                                 G &calculate_local_quantity)
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

} //grid_tools

#endif
