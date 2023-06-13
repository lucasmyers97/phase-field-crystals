#include <iostream>
#include <vector>

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include "parameters/toml.hpp"
#include "utilities/vector_conversion.hpp"

void print_double(double num)
{
    std::cout << num << "\n";
}

int main(int argc, char** argv)
{
    toml::table tbl;
    try
    {
        tbl = toml::parse_file(argv[1]);

        const auto dim = tbl["dim"].value<int>();
        const auto degree = tbl["degree"].value<unsigned int>();

        const auto eps = tbl["eps"].value<double>();

        const auto dt = tbl["dt"].value<double>();
        const auto theta = tbl["theta"].value<double>();
        const auto simulation_tol = tbl["simulation_tol"].value<double>();
        const auto simulation_max_iters = tbl["simulation_max_iters"].value<unsigned int>();

        const auto n_refines = tbl["n_refines"].value<unsigned int>();

        if (!dim) throw std::invalid_argument("No dim in parameter file");
        if (!degree) throw std::invalid_argument("No degree in parameter file");
        if (!eps) throw std::invalid_argument("No eps in parameter file");
        if (!dt) throw std::invalid_argument("No dt in parameter file");
        if (!theta) throw std::invalid_argument("No theta in parameter file");
        if (!simulation_tol) throw std::invalid_argument("No simulation_tol in parameter file");
        if (!simulation_max_iters) throw std::invalid_argument("No simulation_max_iters in parameter file");
        if (!n_refines) throw std::invalid_argument("No n_refines in parameter file");

        dealii::Point<2> p1;
        if (toml::array* p1_array = tbl["p1"].as_array())
        {
            p1 = vector_conversion::convert<dealii::Point<2>>( 
                    toml::convert<std::vector<double>>(*p1_array)
                    );
        }
        else 
        {
            throw std::invalid_argument("p1 is not an array!");
        }

        dealii::Point<2> p2;
        if (toml::array* p2_array = tbl["p2"].as_array())
        {
            p2 = vector_conversion::convert<dealii::Point<2>>( 
                    toml::convert<std::vector<double>>(*p2_array)
                    );
        }
        else 
        {
            throw std::invalid_argument("p1 is not an array!");
        }

        const auto scale_by_lattice_constant = tbl["scale_by_lattice_constant"].value<bool>();

        const double a = 4 * M_PI / std::sqrt(3);
        if (scale_by_lattice_constant.value())
        {
            p1 *= a;
            p2 *= a;
        }

        const auto psi_0 = tbl["psi_0"].value<double>();
        double A_0 = 0;
        if (tbl["A_0"].is_number())
            A_0 = tbl["A_0"].value<double>().value();
        else if (tbl["A_0"].is_string() 
                 && tbl["A_0"].value<std::string>().value() == "default")
            A_0 = 0.2 * (std::abs(*psi_0)
                    + (1.0 / 3.0) * std::sqrt(-15 * *eps - 36 * (*psi_0) * (*psi_0)));
        else
            throw std::invalid_argument("Incorrect input for A_0");

        std::vector<dealii::Tensor<1, 2>> dislocation_positions;
        if (toml::array* dislocation_positions_array = tbl["dislocation_positions"].as_array())
        {
            dislocation_positions 
                = vector_conversion::convert<std::vector<dealii::Tensor<1, 2>>>(
                        toml::convert<std::vector<std::vector<double>>>(
                            *dislocation_positions_array
                        )
                    );
        }
        else
        {
            throw std::invalid_argument("Incorrect input for dislocation positions");
        }

        if (scale_by_lattice_constant.value())
            for (auto& dislocation_position : dislocation_positions)
                dislocation_position *= a;

        std::vector<dealii::Tensor<1, 2>> burgers_vectors;
        if (toml::array* array = tbl["burgers_vectors"].as_array())
        {
            burgers_vectors 
                = vector_conversion::convert<std::vector<dealii::Tensor<1, 2>>>(
                        toml::convert<std::vector<std::vector<double>>>(
                            *array
                        )
                    );
        }
        else
        {
            throw std::invalid_argument("Incorrect input for burgers vectors");
        }

        if (scale_by_lattice_constant.value())
            for (auto& burgers_vector : burgers_vectors)
                burgers_vector *= a;

        std::cout << p1 << "\n";
        std::cout << p2 << "\n";

        for (const auto& burgers_vector : burgers_vectors)
            std::cout << burgers_vector << "\n";

        for (const auto& dislocation_position : dislocation_positions)
            std::cout << dislocation_position << "\n";
     }
     catch (const toml::parse_error& err)
     {
         std::cerr << "Parsing failed:\n" << err << "\n";
         return 1;
     }

    return 0;
}
