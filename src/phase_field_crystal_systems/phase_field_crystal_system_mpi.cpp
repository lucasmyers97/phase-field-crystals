#include "phase_field_crystal_system_mpi.hpp"

#include <deal.II/base/mpi.h>

#include <deal.II/base/array_view.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/types.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/grid/tria.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/numerics/data_component_interpretation.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/vector_tools_interpolate.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>
#include <string>

#include "phase_field_functions/hexagonal_lattice.hpp"

class PsiMatrix : public dealii::Subscriptor
{
public:
    PsiMatrix(const dealii::SparseMatrix<double> &B,
              const dealii::SparseMatrix<double> &C,
              const dealii::SparseMatrix<double> &D,
              const dealii::SparseMatrix<double> &L_psi,
              const dealii::SparseMatrix<double> &L_chi,
              const dealii::SparseDirectUMFPACK &M_chi_inv,
              const dealii::SparseDirectUMFPACK &M_phi_inv)
        : B(B)
        , C(C)
        , D(D)
        , L_psi(L_psi)
        , L_chi(L_chi)
        , M_chi_inv(M_chi_inv)
        , M_phi_inv(M_phi_inv)
    {};

    void vmult(dealii::Vector<double> &dst, dealii::Vector<double> &src) const;

private:
    const dealii::SparseMatrix<double> &B;
    const dealii::SparseMatrix<double> &C;
    const dealii::SparseMatrix<double> &D;
    const dealii::SparseMatrix<double> &L_psi;
    const dealii::SparseMatrix<double> &L_chi;
    const dealii::SparseDirectUMFPACK &M_chi_inv;
    const dealii::SparseDirectUMFPACK &M_phi_inv;
};



void PsiMatrix::vmult(dealii::Vector<double> &dst, dealii::Vector<double> &src) const
{
    dealii::Vector<double> tmp_psi(src.size());
    dealii::Vector<double> tmp_chi_1(L_psi.m());
    dealii::Vector<double> tmp_chi_2(L_psi.m());
    dealii::Vector<double> tmp_phi_1(L_chi.m());
    dealii::Vector<double> tmp_phi_2(L_chi.m());

    L_psi.vmult(tmp_chi_1, src);
    M_chi_inv.vmult(tmp_chi_2, tmp_chi_1);

    C.vmult(tmp_psi, tmp_chi_2);

    L_chi.vmult(tmp_phi_1, tmp_chi_2);
    M_phi_inv.vmult(tmp_phi_2, tmp_phi_1);
    D.vmult(dst, tmp_phi_2);

    dst -= tmp_psi;

    B.vmult(tmp_psi, src);

    dst += tmp_psi;
}



template <int dim>
PhaseFieldCrystalSystemMPI<dim>::PhaseFieldCrystalSystemMPI(unsigned int degree)
    : mpi_communicator(MPI_COMM_WORLD)
    , triangulation(mpi_communicator,
                    typename dealii::Triangulation<dim>::MeshSmoothing(
                    dealii::Triangulation<dim>::smoothing_on_refinement |
                    dealii::Triangulation<dim>::smoothing_on_coarsening))
    , fe_system(dealii::FE_Q<dim>(degree), 1,
                dealii::FE_Q<dim>(degree), 1,
                dealii::FE_Q<dim>(degree), 1)
    , dof_handler(triangulation)
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , timer(mpi_communicator,
            pcout,
            dealii::TimerOutput::never,
            dealii::TimerOutput::wall_times)
{}



template <int dim>
void PhaseFieldCrystalSystemMPI<dim>::make_grid(unsigned int n_refines)
{
    const double a = 4 * M_PI / std::sqrt(3);

    // const int n_across = 5;
    // const int n_down = 5;
    // const double left = -n_across * (2 / std::sqrt(3)) * M_PI * 2;
    // const double down = -n_down * 2 * M_PI;
    // const double left = -20 * a;
    // const double down = -20 * std::sqrt(3) * a;
    const double left = -6 * a;
    const double down = -5 * a;

    dealii::Point<dim> p1 = {left, down};
    dealii::Point<dim> p2 = -p1;

    dealii::GridGenerator::hyper_rectangle(triangulation, 
                                           p1, 
                                           p2, 
                                           /*colorize*/ true);

    using PeriodicFaces
        = std::vector<dealii::GridTools::PeriodicFacePair<
            typename dealii::parallel::distributed::Triangulation<dim>::cell_iterator
                >
            >;

    PeriodicFaces x_periodic_faces;
    PeriodicFaces y_periodic_faces;
    dealii::GridTools::collect_periodic_faces(triangulation,
                                              /*b_id1*/ 0,
                                              /*b_id2*/ 1,
                                              /*direction*/ 0,
                                              x_periodic_faces);
    dealii::GridTools::collect_periodic_faces(triangulation,
                                              /*b_id1*/ 2,
                                              /*b_id2*/ 3,
                                              /*direction*/ 1,
                                              y_periodic_faces);

    triangulation.add_periodicity(x_periodic_faces);
    triangulation.add_periodicity(y_periodic_faces);
    triangulation.refine_global(n_refines);
}



template <int dim>
void PhaseFieldCrystalSystemMPI<dim>::setup_dofs()
{
    dof_handler.distribute_dofs(fe_system);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    dealii::DoFTools::extract_locally_relevant_dofs(dof_handler,
                                                    locally_relevant_dofs);

    std::vector<unsigned int> block_component = {0, 1, 2};
    dealii::DoFRenumbering::component_wise(dof_handler, block_component);
    {
        constraints.clear();
        constraints.reinit(locally_relevant_dofs);
        dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);

        using PeriodicFaces
            = std::vector<dealii::GridTools::PeriodicFacePair<
                typename dealii::DoFHandler<dim>::cell_iterator
                    >
                >;

        PeriodicFaces x_periodic_faces;
        PeriodicFaces y_periodic_faces;
        dealii::GridTools::collect_periodic_faces(dof_handler,
                                                  /*b_id1*/ 0,
                                                  /*b_id2*/ 1,
                                                  /*direction*/ 0,
                                                  x_periodic_faces);
        dealii::GridTools::collect_periodic_faces(dof_handler,
                                                  /*b_id1*/ 2,
                                                  /*b_id2*/ 3,
                                                  /*direction*/ 1,
                                                  y_periodic_faces);

        dealii::DoFTools::
            make_periodicity_constraints<dim, dim>(x_periodic_faces,
                                                   constraints);
        dealii::DoFTools::
            make_periodicity_constraints<dim, dim>(y_periodic_faces,
                                                   constraints);
        constraints.close();
    }
    {
        dealii::DynamicSparsityPattern dsp(locally_relevant_dofs);
        dealii::DoFTools::make_sparsity_pattern(dof_handler, 
                                                dsp, 
                                                constraints, 
                                                /*keep_constrained_dofs*/false);
        dealii::SparsityTools::distribute_sparsity_pattern(
            dsp,
            dof_handler.locally_owned_dofs(),
            mpi_communicator,
            locally_relevant_dofs);

        system_matrix.reinit(locally_owned_dofs,
                             locally_owned_dofs,
                             dsp,
                             mpi_communicator);
    }

    system_rhs.reinit(locally_owned_dofs, mpi_communicator);
}



template <int dim>
void PhaseFieldCrystalSystemMPI<dim>::initialize_fe_field()
{
    double psi_0 = -0.43;
    double A_0 = 0.2 * (std::abs(psi_0)
                        + (1.0 / 3.0) * std::sqrt(-15 * eps - 36 * psi_0 * psi_0));

    double a = 4 * M_PI / std::sqrt(3);

    std::vector<dealii::Tensor<1, dim>> dislocation_positions;
    // dislocation_positions.push_back(dealii::Tensor<1, dim>({2 * a, 0}));
    // dislocation_positions.push_back(dealii::Tensor<1, dim>({-2 * a, 0}));

    std::vector<dealii::Tensor<1, dim>> burgers_vectors;
    // burgers_vectors.push_back(dealii::Tensor<1, dim>({a, 0}));
    // burgers_vectors.push_back(dealii::Tensor<1, dim>({-a, 0}));

    HexagonalLattice<dim> hexagonal_lattice(A_0, 
                                            psi_0, 
                                            dislocation_positions, 
                                            burgers_vectors);

    dealii::LinearAlgebraTrilinos::MPI::Vector Psi_0(locally_owned_dofs, 
                                                     mpi_communicator);
    dealii::VectorTools::interpolate(dof_handler,
                                     hexagonal_lattice,
                                     Psi_0);
    constraints.distribute(Psi_0);
    Psi_0.compress(dealii::VectorOperation::insert);

    Psi_n.reinit(locally_owned_dofs, 
                 locally_relevant_dofs, 
                 mpi_communicator);
    Psi_n_1.reinit(locally_owned_dofs, 
                   locally_relevant_dofs, 
                   mpi_communicator);

    Psi_n = Psi_0;
    Psi_n_1 = Psi_0;
}



template <int dim>
void PhaseFieldCrystalSystemMPI<dim>::assemble_system()
{
    system_matrix = 0;
    system_rhs = 0;

    dealii::QGauss<dim> quadrature_formula(fe_system.degree + 1);
    
    dealii::FEValues<dim> fe_values(fe_system,
                                    quadrature_formula,
                                    dealii::update_values |
                                    dealii::update_gradients |
                                    dealii::update_quadrature_points |
                                    dealii::update_JxW_values);

    const unsigned int dofs_per_cell = fe_system.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();

    dealii::FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    dealii::Vector<double> local_rhs(dofs_per_cell);

    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    const dealii::FEValuesExtractors::Scalar psi(0);
    const dealii::FEValuesExtractors::Scalar chi(1);
    const dealii::FEValuesExtractors::Scalar phi(2);

    std::vector<double> eta_psi(dofs_per_cell);
    std::vector<double> eta_chi(dofs_per_cell);
    std::vector<double> eta_phi(dofs_per_cell);
    std::vector<dealii::Tensor<1, dim>> grad_eta_psi(dofs_per_cell);
    std::vector<dealii::Tensor<1, dim>> grad_eta_chi(dofs_per_cell);
    std::vector<dealii::Tensor<1, dim>> grad_eta_phi(dofs_per_cell);

    std::vector<double> psi_n(n_q_points);
    std::vector<double> chi_n(n_q_points);
    std::vector<double> phi_n(n_q_points);
    std::vector<double> psi_n_1(n_q_points);
    std::vector<double> chi_n_1(n_q_points);
    std::vector<double> phi_n_1(n_q_points);
    std::vector<dealii::Tensor<1, dim>> grad_psi_n(n_q_points);
    std::vector<dealii::Tensor<1, dim>> grad_chi_n(n_q_points);
    std::vector<dealii::Tensor<1, dim>> grad_phi_n(n_q_points);
    std::vector<dealii::Tensor<1, dim>> grad_psi_n_1(n_q_points);
    std::vector<dealii::Tensor<1, dim>> grad_chi_n_1(n_q_points);
    std::vector<dealii::Tensor<1, dim>> grad_phi_n_1(n_q_points);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (!(cell->is_locally_owned()))
            continue;

        fe_values.reinit(cell);
        local_matrix = 0;
        local_rhs = 0;

        for (const unsigned int q : fe_values.quadrature_point_indices())
        {
            fe_values[psi].get_function_values(Psi_n, psi_n);
            fe_values[chi].get_function_values(Psi_n, chi_n);
            fe_values[phi].get_function_values(Psi_n, phi_n);
            fe_values[psi].get_function_values(Psi_n_1, psi_n_1);
            fe_values[chi].get_function_values(Psi_n_1, chi_n_1);
            fe_values[phi].get_function_values(Psi_n_1, phi_n_1);
            fe_values[psi].get_function_gradients(Psi_n, grad_psi_n);
            fe_values[chi].get_function_gradients(Psi_n, grad_chi_n);
            fe_values[phi].get_function_gradients(Psi_n, grad_phi_n);
            fe_values[psi].get_function_gradients(Psi_n_1, grad_psi_n_1);
            fe_values[chi].get_function_gradients(Psi_n_1, grad_chi_n_1);
            fe_values[phi].get_function_gradients(Psi_n_1, grad_phi_n_1);

            for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
                grad_eta_psi[k] = fe_values[psi].gradient(k, q);
                grad_eta_chi[k] = fe_values[chi].gradient(k, q);
                grad_eta_phi[k] = fe_values[phi].gradient(k, q);
                eta_psi[k] = fe_values[psi].value(k, q);
                eta_chi[k] = fe_values[chi].value(k, q);
                eta_phi[k] = fe_values[phi].value(k, q);
            }

            for (const unsigned int i : fe_values.dof_indices())
            {
                local_rhs(i) +=
                    (
                     (eta_psi[i]
                      * (-psi_n[q] + psi_n_1[q]
                         + dt * theta * (2 * phi_n[q]
                                         + (1 + eps) * chi_n[q]
                                         + 3 * psi_n[q] * psi_n[q] * chi_n[q]
                                         + 6 * psi_n[q] * grad_psi_n[q] * grad_psi_n[q]
                             )
                         + dt * (1 - theta) * (2 * phi_n_1[q]
                                               + (1 + eps) * chi_n_1[q]
                                               + 3 * psi_n_1[q] * psi_n_1[q] * chi_n_1[q]
                                               + 6 * psi_n_1[q] * grad_psi_n_1[q] * grad_psi_n_1[q]
                             )
                      )
                     )
                     -
                     (dt * grad_eta_psi[i]
                      * (theta * grad_phi_n[q] + (1 - theta) * grad_phi_n_1[q])
                     )
                     -
                     (eta_chi[i] * chi_n[q])
                     -
                     (grad_eta_chi[i] * grad_psi_n[q])
                     -
                     (eta_phi[i] * phi_n[q])
                     -
                     (grad_eta_phi[i] * grad_chi_n[q])
                    )
                    * fe_values.JxW(q);

                for (const unsigned int j : fe_values.dof_indices())
                {
                    local_matrix(i, j) +=
                        (
                         (eta_psi[i] 
                          * (eta_psi[j] 
                             - dt * theta * (2 * eta_phi[j]
                                             + (1 + eps) * eta_chi[j]
                                             + 3 * psi_n[q] * psi_n[q] * eta_chi[j]
                                             + 6 * psi_n[q] * chi_n[q] * eta_psi[j]
                                             + 6 * grad_psi_n[q] * grad_psi_n[q] * eta_psi[j]
                                             + 12 * psi_n[q] * grad_psi_n[q] * grad_eta_psi[j])
                            )
                         )
                         +
                         (dt * theta * grad_eta_psi[i] * grad_eta_phi[j])
                         +
                         (eta_chi[i] * eta_chi[j])
                         +
                         (grad_eta_chi[i] * grad_eta_psi[j])
                         +
                         (eta_phi[i] * eta_phi[j])
                         +
                         (grad_eta_phi[i] * grad_eta_chi[j])
                        )
                        * fe_values.JxW(q); 
                }
            }
        }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(local_matrix,
                                               local_rhs,
                                               local_dof_indices,
                                               system_matrix,
                                               system_rhs);
    }

    system_matrix.compress(dealii::VectorOperation::add);
    system_rhs.compress(dealii::VectorOperation::add);
}



template <int dim>
void PhaseFieldCrystalSystemMPI<dim>::solve_and_update()
{
    dealii::LinearAlgebraTrilinos::MPI::PreconditionAMG::AdditionalData data;
    data.elliptic = false;

    // dealii::LinearAlgebraTrilinos::MPI::PreconditionAMG preconditioner;
    // preconditioner.initialize(system_matrix, data);
    // dealii::TrilinosWrappers::PreconditionIdentity preconditioner;
    dealii::LinearAlgebraTrilinos::MPI::PreconditionJacobi preconditioner;
    // dealii::LinearAlgebraTrilinos::MPI::PreconditionILU preconditioner;
    preconditioner.initialize(system_matrix);

    dealii::SolverControl solver_control(dof_handler.n_dofs());
    dealii::SolverGMRES<dealii::LinearAlgebraTrilinos::MPI::Vector>
        solver_gmres(solver_control);

    dealii::LinearAlgebraTrilinos::MPI::Vector dPsi_n(locally_owned_dofs, 
                                                      mpi_communicator);
    solver_gmres.solve(system_matrix,
                       dPsi_n,
                       system_rhs,
                       preconditioner);
    pcout << "Number of iterations: " << solver_control.last_step() << "\n\n";
    constraints.distribute(dPsi_n);

    dealii::LinearAlgebraTrilinos::MPI::Vector local_Psi_n(locally_owned_dofs, 
                                                           mpi_communicator);
    local_Psi_n = Psi_n;
    local_Psi_n += dPsi_n;
    Psi_n = local_Psi_n;
}



template <int dim>
void PhaseFieldCrystalSystemMPI<dim>::iterate_timestep()
{
    for (unsigned int n_iters = 0; n_iters < simulation_max_iters; ++n_iters)
    {
        pcout << "Assembling system!\n";
        assemble_system();

        const double residual = system_rhs.l2_norm();
        pcout << "Residual is: " << residual << "\n";
        if (residual < simulation_tol)
            break;

        pcout << "Solving and updating!\n";
        solve_and_update();
    }
    Psi_n_1 = Psi_n;
}



template <int dim>
void PhaseFieldCrystalSystemMPI<dim>::output_configuration(unsigned int iteration)
{
    std::vector<std::string> field_names = {"psi", "chi", "phi"};
    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(3, dealii::DataComponentInterpretation::component_is_scalar);

    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(Psi_n,
                             field_names,
                             dealii::DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    data_out.build_patches();

    data_out.write_vtu_with_pvtu_record("./", "phase_field_", iteration,
                                        mpi_communicator,
                                        /*n_digits_for_counter*/2);
}



template <int dim>
void PhaseFieldCrystalSystemMPI<dim>::output_rhs(unsigned int iteration)
{
    std::vector<std::string> field_names = {"psi", "chi", "phi"};
    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(3, dealii::DataComponentInterpretation::component_is_scalar);

    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(system_rhs,
                             field_names,
                             dealii::DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    data_out.build_patches();

    data_out.write_vtu_with_pvtu_record("./", "phase_field_rhs_", iteration,
                                        mpi_communicator,
                                        /*n_digits_for_counter*/2);
}



template <int dim>
void PhaseFieldCrystalSystemMPI<dim>::run(unsigned int n_refines)
{
    pcout << "Making grid!\n";
    make_grid(n_refines);

    pcout << "Setting up dofs!\n";
    setup_dofs();

    pcout << "Initializing fe field!\n";
    initialize_fe_field();

    const unsigned int n_timesteps = 10000;
    for (unsigned int timestep = 0; timestep < n_timesteps; ++timestep)
    {
        pcout << "Outputting configuration!\n";
        output_configuration(timestep);

        pcout << "Outputting right-hand side!\n";
        output_rhs(timestep);


        pcout << "\n";
        pcout << "Iterating timestep: " << timestep << "\n";
        iterate_timestep();
    }
}


template class PhaseFieldCrystalSystemMPI<2>;
template class PhaseFieldCrystalSystemMPI<3>;
