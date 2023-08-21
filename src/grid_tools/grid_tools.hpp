#ifndef PHASE_FIELD_CRYSTALS_GRID_TOOLS_HPP
#define PHASE_FIELD_CRYSTALS_GRID_TOOLS_HPP

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>

namespace grid_tools {
   
template <int dim>
using cell_iterator = typename dealii::DoFHandler<dim>::active_cell_iterator;

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
template <int dim>
void visit_neighborhood(dealii::Triangulation<dim>& tria,
                        const cell_iterator<dim>& base_cell, 
                        std::function<bool(const cell_iterator<dim>&)> &is_in_neighborhood,
                        std::function<void(const cell_iterator<dim>&)> &calculate_local_quantity,
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
template <int dim>
void visit_neighbors_recursively(const cell_iterator<dim>& cell, 
                                 std::function<bool(const cell_iterator<dim>&)> &is_in_neighborhood,
                                 std::function<void(const cell_iterator<dim>&)> &calculate_local_quantity);

} //grid_tools

#endif
