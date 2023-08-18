#include <iostream>
#include <fstream>
#include <string>

#include <deal.II/base/point.h>
#include <deal.II/lac/vector.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_update_flags.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/numerics/data_out.h>

constexpr int dim = 2;
constexpr unsigned int n_refines = 3;
constexpr unsigned int degree = 1;
const std::string quad_filename = "neighborhood_test.csv";
const std::string output_filename = "neighborhood_test.vtu";
const double radius = 0.2;

int main()
{
    dealii::Triangulation<dim> tria;
    dealii::GridGenerator::hyper_cube(tria);
    tria.refine_global(n_refines);

    dealii::FE_Q<dim> fe(degree);
    dealii::DoFHandler<dim> dof_handler(tria);
    dof_handler.distribute_dofs(fe);


    dealii::QGauss<dim> quadrature_formula(degree + 1);
    dealii::FEValues fe_values(fe,
                               quadrature_formula,
                               dealii::update_quadrature_points);

    std::ofstream output_file(quad_filename);
    dealii::Vector<float> subdomain(tria.n_active_cells());
    unsigned int i = 0;
    dealii::Point<dim> origin(0.5, 0.5);
    output_file << "x, y, z\n";

    auto is_in_neighborhood = [origin, &fe_values](const typename dealii::DoFHandler<dim>::active_cell_iterator& cell){
        fe_values.reinit(cell);
        const auto& quad_points = fe_values.get_quadrature_points();
        bool in_neighborhood = false;
        for (const auto& point : quad_points)
            in_neighborhood = in_neighborhood || (origin.distance(point) < radius);
        
        return in_neighborhood;
    };
    for (const auto& cell : dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);   

        const auto& quad_points = fe_values.get_quadrature_points();

        for (const auto& point : quad_points)
            output_file << point[0] << ", " << point[1] << ", 0\n";

        // subdomain(i) = cell->center().distance(origin);
        subdomain(i) = is_in_neighborhood(cell) ? 1.0 : 0;
        ++i;
    }

    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(subdomain, "subdomain");
    data_out.build_patches();

    std::ofstream output(output_filename);
    data_out.write_vtu(output);

    return 0;
}
