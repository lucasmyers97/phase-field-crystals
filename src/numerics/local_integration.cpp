#include "local_integration.hpp"

#include "grid_tools/grid_tools.hpp"

#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_handler.h>

#include <functional>
#include <vector>
#include <cmath>

namespace local_integration
{

template <int dim>
using tria_cell_iterator = typename dealii::Triangulation<dim>::active_cell_iterator;

template <int dim>
using cell_iterator = typename dealii::DoFHandler<dim>::active_cell_iterator;

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
                          dealii::SparseMatrix<double> &mass_matrix)
{
    mass_matrix = 0;
    convolved_field_rhs = 0;

    const unsigned int dofs_per_cell = fe_values.dofs_per_cell();

    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);
    dealii::FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    dealii::Vector<double> local_rhs(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        local_matrix = 0;
        local_rhs = 0;

        const dealii::Point<dim> q_points = fe_values.get_quadrature_points();
        for (const unsigned int q : fe_values.quadrature_point_indices())
        {
            double convolution_value 
                = gaussian_convolution_on_neighborhood(tria,
                                                       fe_values_convolution,
                                                       cell, 
                                                       q_points[q],
                                                       sigma,
                                                       integral_radius,
                                                       fe_field);

            for (const unsigned int i : fe_values.dof_indices())
            {
                local_rhs(i) += fe_values.shape_value(i, q)
                                * convolution_value
                                * fe_values.JxW(q);

                for (const unsigned int j : fe_values.dof_indices())
                    local_matrix(i, j) += fe_values.shape_value(i, q)
                                          * fe_values.shape_value(j, q)
                                          * fe_values.JxW(q);
            }
        }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(local_matrix,
                                               local_rhs,
                                               local_dof_indices,
                                               mass_matrix,
                                               convolved_field_rhs);
    }

    convolved_field_rhs *= 1 / (2 * dealii::numbers::PI * sigma * sigma);
}



template <int dim>
double gaussian_convolution_on_neighborhood(dealii::Triangulation<dim>& tria,
                                            const dealii::DoFHandler<dim>& dof_handler,
                                            const dealii::FEValues<dim>& fe_values,
                                            const cell_iterator<dim>& base_cell, 
                                            const dealii::Point<dim>& base_point,
                                            double sigma,
                                            double integral_radius,
                                            const dealii::Vector<double> &fe_field)
{
    std::function<bool(const tria_cell_iterator<dim>&)> is_in_neighborhood 
        = grid_tools::neighborhood_functions::IsInL2Neighborhood<dim>(fe_values, base_point, integral_radius);

    double integral_result = 0;
    std::function<void(const tria_cell_iterator<dim>&)> calculate_cell_integral_contribution
        = [&fe_field, &fe_values, &dof_handler, &base_point, sigma, integral_radius, &integral_result]
        (const tria_cell_iterator<dim>& tria_cell) {
            cell_iterator<dim> cell(&tria_cell->get_triangulation(),
                                    tria_cell->level(),
                                    tria_cell->index(),
                                    &dof_handler);
            fe_values.reinit(cell);

            const std::vector<dealii::Point<dim>> q_points = fe_values.get_quadrature_points();
            std::vector<double> fe_field_values(fe_values.n_quadrature_points);
            fe_values.get_function_values(fe_field);
            for (const unsigned int q : fe_values.quadrature_point_indices())
            {
                integral_result += fe_field_values[q]
                                   * std::exp(-(base_point - q_points[q]).square() / (2 * sigma * sigma))
                                   * fe_values.JxW(q);
            }
        };

    grid_tools::visit_neighborhood<dim>(tria, 
                                        base_cell, 
                                        is_in_neighborhood,
                                        calculate_cell_integral_contribution);

    return integral_result;
}



template <typename T, int dim>
void locally_integrate(std::function<T (dealii::Point<dim>&)> &integrand,
                    const dealii::DoFHandler<dim> &dof_handler,
                    const typename dealii::DoFHandler<dim>::cell_accessor &cell,
                    const dealii::Point<dim> &center,
                    double radius)
{}



template <int dim>
std::vector<typename dealii::DoFHandler<dim>::cell_iterator>
find_cells_in_distance(const dealii::DoFHandler<dim> &dof_handler,
                       const typename dealii::DoFHandler<dim>::cell_iterator center_cell,
                       const dealii::Point<dim> &center,
                       double radius)
{
    const dealii::FiniteElement<dim> &fe = dof_handler.get_fe();
    dealii::QGauss<dim> q(fe.degree + 1);
    dealii::FEValues<dim> fe_values(fe, q, dealii::update_quadrature_points);

    std::vector<dealii::Point<dim>> quad_points(q.size());

    std::vector<typename dealii::DoFHandler<dim>::cell_iterator> cell_list = {center_cell};
    for (std::size_t cell_no = 0; cell_no < cell_list.size(); ++cell_no)
    {
        const auto cell = cell_list[cell_no];

        const unsigned int n_faces = cell->n_faces();
        for (unsigned int face_no = 0; face_no < n_faces; ++face_no)
        {
            if (cell->at_boundary(face_no))
                continue;

            auto neighbor = cell->neighbor(face_no);
            if (!neighbor->has_children())
            {
                if (neighbor->is_artificial() || neighbor->user_flag_set())
                    continue;

                fe_values.reinit(neighbor);
                quad_points = fe_values.get_quadrature_points();
                for (const auto& quad_point : quad_points)
                    if (quad_point.distance(center) < radius)
                    {
                        cell_list.push_back(neighbor);
                        neighbor->set_user_flag();
                        break;
                    }
            }
            else 
            {
                for (unsigned int child_no = 0; child_no < neighbor->n_children(); ++child_no)
                {
                    auto child = neighbor->child(child_no);
                    if (child->is_artificial() || child->user_flag_set())
                        continue;

                    fe_values.reinit(child);
                    quad_points = fe_values.get_quadrature_points();
                    for (const auto& quad_point : quad_points)
                        if (quad_point.distance(center) < radius)
                        {
                            cell_list.push_back(child);
                            child->set_user_flag();
                            break;
                        }
                }
            }
        }
    }

    return cell_list;
}

template
std::vector<typename dealii::DoFHandler<2>::cell_iterator>
find_cells_in_distance(const dealii::DoFHandler<2> &dof_handler,
                       const typename dealii::DoFHandler<2>::cell_iterator center_cell,
                       const dealii::Point<2> &center,
                       double radius);

template
std::vector<typename dealii::DoFHandler<3>::cell_iterator>
find_cells_in_distance(const dealii::DoFHandler<3> &dof_handler,
                       const typename dealii::DoFHandler<3>::cell_iterator center_cell,
                       const dealii::Point<3> &center,
                       double radius);

} // local_integration
