#include "local_integration.hpp"

#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_handler.h>

#include <functional>
#include <vector>

namespace local_integration
{

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
