#ifndef LOCAL_INTEGRATION_HPP
#define LOCAL_INTEGRATION_HPP

#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>

#include <functional>

namespace local_integration
{

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

} // local_integration

#endif
