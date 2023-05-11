#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <memory>
#include <utility>
#include <string>
#include <filesystem>

#include "phase_field_functions/hexagonal_lattice.hpp"
#include "phase_field_crystal_systems/phase_field_crystal_system_mpi.hpp"
#include "parameters/toml.hpp"

template <int dim>
std::unique_ptr<PhaseFieldCrystalSystemMPI<dim>>
parse_simulation_parameters(const toml::table& tbl)
{
    // filesystem parameters
    const auto data_folder = tbl["data_folder"].value<std::string>();
    const auto configuration_filename = tbl["configuration_filename"].value<std::string>();
    const auto rhs_filename = tbl["rhs_filename"].value<std::string>();

    // throw excetion for missing parameters
    if (!data_folder) throw std::invalid_argument("No data folder in parameter file");
    if (!configuration_filename) throw std::invalid_argument("No configuration filename in parameter file");
    if (!rhs_filename) throw std::invalid_argument("No rhs filename in parameter file");

    const auto data_folder_fs = std::filesystem::path(data_folder.value());
    const auto configuration_filename_fs = std::filesystem::path(configuration_filename.value());
    const auto rhs_filename_fs = std::filesystem::path(rhs_filename.value());

    // simulation parameters
    const auto degree = tbl["degree"].value<unsigned int>();
    const auto eps = tbl["eps"].value<double>();
    const auto dt = tbl["dt"].value<double>();
    const auto n_timesteps = tbl["n_timesteps"].value<unsigned int>();
    const auto theta = tbl["theta"].value<double>();
    const auto simulation_tol = tbl["simulation_tol"].value<double>();
    const auto simulation_max_iters = tbl["simulation_max_iters"].value<unsigned int>();
    const auto n_refines = tbl["n_refines"].value<unsigned int>();

    // if there are missing parameters, throw exception
    if (!degree) throw std::invalid_argument("No degree in parameter file");
    if (!eps) throw std::invalid_argument("No eps in parameter file");
    if (!dt) throw std::invalid_argument("No dt in parameter file");
    if (!n_timesteps) throw std::invalid_argument("No n_timesteps in parameter file");
    if (!theta) throw std::invalid_argument("No theta in parameter file");
    if (!simulation_tol) throw std::invalid_argument("No simulation_tol in parameter file");
    if (!simulation_max_iters) throw std::invalid_argument("No simulation_max_iters in parameter file");
    if (!n_refines) throw std::invalid_argument("No n_refines in parameter file");

    // grid parameters
    dealii::Point<dim> p1;
    if (const toml::array* p1_array = tbl["p1"].as_array())
        p1 = toml::convert<dealii::Point<dim>>(*p1_array);
    else 
        throw std::invalid_argument("p1 is not an array!");

    dealii::Point<dim> p2;
    if (const toml::array* p2_array = tbl["p2"].as_array())
        p2 = toml::convert<dealii::Point<dim>>(*p2_array);
    else 
        throw std::invalid_argument("p1 is not an array!");

    double a = HexagonalLattice<dim>::a;
    const auto scale_by_lattice_constant = tbl["scale_by_lattice_constant"].value<bool>();

    if (scale_by_lattice_constant.value())
    {
        p1 *= a;
        p2 *= a;
    }

    // initial_condition parameters
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

    std::vector<dealii::Tensor<1, dim>> dislocation_positions;
    if (const toml::array* array = tbl["dislocation_positions"].as_array())
        dislocation_positions = toml::convert<std::vector<dealii::Tensor<1, dim>>>(*array);
    else
        throw std::invalid_argument("Incorrect input for dislocation positions");

    if (scale_by_lattice_constant.value())
        for (auto& dislocation_position : dislocation_positions)
            dislocation_position *= a;

    std::vector<dealii::Tensor<1, dim>> burgers_vectors;
    if (const toml::array* array = tbl["burgers_vectors"].as_array())
        burgers_vectors = toml::convert<std::vector<dealii::Tensor<1, dim>>>(*array);
    else
        throw std::invalid_argument("Incorrect input for burgers vectors");

    if (scale_by_lattice_constant.value())
        for (auto& burgers_vector : burgers_vectors)
            burgers_vector *= a;

    std::unique_ptr<dealii::Function<dim>> initial_condition 
        = std::make_unique<HexagonalLattice<dim>>(A_0, 
                                                  psi_0.value(), 
                                                  dislocation_positions, 
                                                  burgers_vectors);

    std::unique_ptr<PhaseFieldCrystalSystemMPI<dim>> phase_field_crystal_system_mpi
        = std::make_unique<PhaseFieldCrystalSystemMPI<dim>>(degree.value(),
                                                            data_folder_fs,
                                                            configuration_filename_fs,
                                                            rhs_filename_fs,
                                                            eps.value(),
                                                            dt.value(),
                                                            n_timesteps.value(),
                                                            theta.value(),
                                                            simulation_tol.value(),
                                                            simulation_max_iters.value(),
                                                            n_refines.value(),
                                                            p1,
                                                            p2,
                                                            std::move(initial_condition));

    return phase_field_crystal_system_mpi;
}


int main(int ac, char* av[])
{
    try
    {
        dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(ac, av, 1);

        toml::table tbl;
        try
        {
            tbl = toml::parse_file(av[1]);
        }
        catch (const toml::parse_error& err)
        {
            std::cerr << "Parsing failed:\n" << err << "\n";
            return 1;
        }

        const auto dim = tbl["dim"].value<int>();
        if (dim.value() == 2)
        {
            auto phase_field_crystal_system_mpi = parse_simulation_parameters<2>(tbl);
            phase_field_crystal_system_mpi->run();
        }
        else if (dim.value() == 3)
        {
            auto phase_field_crystal_system_mpi = parse_simulation_parameters<3>(tbl);
            phase_field_crystal_system_mpi->run();
        }
    }
    catch (const std::exception &err)
    {
        std::cerr << "--------------------------\n";
        std::cerr << "Found error in program!\n";
        std::cerr << err.what();
        std::cerr << "--------------------------\n";
    }

    return 0;
}
