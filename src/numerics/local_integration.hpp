#ifndef LOCAL_INTEGRATION_HPP
#define LOCAL_INTEGRATION_HPP

#include <deal.II/base/point.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_handler.h>

#include <functional>

#include "grid_tools/grid_tools.hpp"

namespace local_integration
{

template <int dim>
using tria_cell_iterator = typename dealii::Triangulation<dim>::active_cell_iterator;

template <int dim>
using cell_iterator = typename dealii::DoFHandler<dim>::active_cell_iterator;

template <int dim, typename T>
using convolution_function = std::function<std::vector<T>(const tria_cell_iterator<dim> &)>;

template <int dim>
void gaussian_convolution(dealii::Triangulation<dim>& tria,
                          const dealii::DoFHandler<dim>& dof_handler,
                          dealii::FEValues<dim> &fe_values,
                          dealii::FEValues<dim> &fe_values_convolution,
                          dealii::AffineConstraints<double> &constraints,
                          double sigma,
                          double integral_radius,
                          const dealii::Vector<double> &fe_field,
                          dealii::Vector<double> &convolved_field_rhs,
                          dealii::SparseMatrix<double> &mass_matrix);

template <int dim>
double gaussian_convolution_on_neighborhood(dealii::Triangulation<dim>& tria,
                                            const dealii::DoFHandler<dim>& dof_handler,
                                            const dealii::FiniteElement<dim>& fe,
                                            const cell_iterator<dim>& base_cell, 
                                            const dealii::Point<dim>& base_point,
                                            double sigma,
                                            double integral_radius,
                                            const dealii::Vector<double> &fe_field);

template <int dim, typename T, typename S, typename R>
R local_convolution_at_point(dealii::Triangulation<dim> &tria,
                             const dealii::DoFHandler<dim> &dof_handler,
                             dealii::FEValues<dim> &fe_values,
                             convolution_function<dim, T> &f,
                             convolution_function<dim, S> &g,
                             std::function<bool(tria_cell_iterator<dim>&)> &is_in_neighborhood,
                             tria_cell_iterator<dim>& base_cell);

template <typename T, int dim>
void locally_integrate(std::function<T (dealii::Point<dim>&)> &integrand,
                    const dealii::DoFHandler<dim> &dof_handler,
                    const typename dealii::DoFHandler<dim>::cell_accessor &cell,
                    const dealii::Point<dim> &center,
                    double radius);



template <int dim>
std::vector<typename dealii::DoFHandler<dim>::cell_iterator>
find_cells_in_distance(const dealii::DoFHandler<dim> &dof_handler,
                       const typename dealii::DoFHandler<dim>::cell_iterator cell,
                       const dealii::Point<dim> &center,
                       double radius);



template <int dim, typename T, typename S, typename R, typename F, typename G>
R local_convolution_at_point(dealii::Triangulation<dim> &tria,
                             const dealii::DoFHandler<dim> &dof_handler,
                             dealii::FEValues<dim> &fe_values,
                             F &f,
                             G &g,
                             std::function<bool(const tria_cell_iterator<dim>&)> &is_in_neighborhood,
                             tria_cell_iterator<dim>& base_cell)
{
    R integral_result = 0;
    std::function<void(const tria_cell_iterator<dim>&)> calculate_cell_integral_contribution
        = [&dof_handler, &fe_values, &f, &g, &integral_result]
        (const tria_cell_iterator<dim>& tria_cell) {
            cell_iterator<dim> cell(&tria_cell->get_triangulation(),
                                    tria_cell->level(),
                                    tria_cell->index(),
                                    &dof_handler);
            fe_values.reinit(cell);

            std::vector<T> f_vals = f(tria_cell);
            std::vector<R> g_vals = g(tria_cell);
            for (const unsigned int q : fe_values.quadrature_point_indices())
            {
                integral_result += f_vals[q]
                                   * g_vals[q]
                                   * fe_values.JxW(q);
            }
        };

    grid_tools::visit_neighborhood<dim>(tria, 
                                        base_cell, 
                                        is_in_neighborhood,
                                        calculate_cell_integral_contribution);

    return integral_result;
}

} // local_integration

#endif
