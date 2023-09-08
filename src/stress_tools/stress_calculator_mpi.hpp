#ifndef STRESS_CALCULATOR_MPI_HPP
#define STRESS_CALCULATOR_MPI_HPP

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/generic_linear_algebra.h>

#include <deal.II/base/mpi.h>

#include <vector>
#include <filesystem>

/**
 * \brief Calculates tensorial stress field associated with the phase field
 *
 * Given a triangulation and degree of Lagrange polynomials, this class
 * creates all of the machinery necessary for calculating the stress tensor 
 * field \f$\sigma\f$ arising from the phase-field \f$\Psi\f$.
 * For sake of efficiency, this calculation is broken up into three steps
 * which all have to be called initially in order to do the operation.
 * These are:
 *
 * - setup_dofs
 * - calculate_mass_matrix
 * - calculate_stress
 *
 * After the initial calculation, only `calculate_stress` needs to be rerun
 * on update of \f$\Psi\f$.
 * setup_dofs needs to be rerun only if the Triangulation is modified, and 
 * calculate_mass_matrix needs to be rerun only if setup_dofs is.
 * Typical usage is:
 * ```cpp
 * StressCalculatorMPI stress_calculator(tria, degree);
 *
 * stress_calculator.setup_dofs(mpi_communicator);
 * stress_calculator.calculate_mass_matrix();
 * stress_calculator.calculate_stress(mpi_communicator,
 *                                    Psi_dof_handler,
 *                                    Psi,
 *                                    eps);
 * ```
 */
template <int dim>
class StressCalculatorMPI
{
public:
    StressCalculatorMPI(const dealii::Triangulation<dim> &tria,
                        const unsigned int degree);
    std::vector<unsigned int> calculate_stress(const MPI_Comm& mpi_communicator,
                                               const dealii::DoFHandler<dim> &Psi_dof_handler,
                                               const dealii::LinearAlgebraTrilinos::MPI::BlockVector &Psi,
                                               double eps);
    void setup_dofs(const MPI_Comm& mpi_communicator);
    void calculate_mass_matrix();
    void output_stress(const MPI_Comm& mpi_communicator,
                       std::filesystem::path data_folder,
                       std::filesystem::path stress_filename,
                       unsigned int iteration);

private:
    void calculate_righthand_side(const dealii::DoFHandler<dim> &Psi_dof_handler,
                                  const dealii::LinearAlgebraTrilinos::MPI::BlockVector &Psi,
                                  double eps);
    std::vector<unsigned int> solve(const MPI_Comm& mpi_communicator);

    dealii::DoFHandler<dim> dof_handler;
    dealii::FESystem<dim> fe_system;
    dealii::AffineConstraints<double> constraints;

    dealii::LinearAlgebraTrilinos::MPI::BlockSparseMatrix system_matrix;
    dealii::LinearAlgebraTrilinos::MPI::BlockVector system_rhs;
    dealii::LinearAlgebraTrilinos::MPI::BlockVector stress;

    std::vector<dealii::IndexSet> owned_partitioning;
    std::vector<dealii::IndexSet> relevant_partitioning;
};

#endif
