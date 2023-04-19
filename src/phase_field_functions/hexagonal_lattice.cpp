#include "hexagonal_lattice.hpp"

#include <deal.II/base/point.h>

#include <cmath>
#include <stdexcept>

template <int dim>
HexagonalLattice<dim>::HexagonalLattice()
    : dealii::Function<dim>(3)
{
    q.resize(3);

    q[0][0] = 0.0;
    q[0][1] = 1.0;

    q[1][0] = 0.5 * std::sqrt(3.0);
    q[1][1] = -0.5;

    q[2][0] = -0.5 * std::sqrt(3.0);
    q[2][1] = -0.5;
}



template <int dim>
HexagonalLattice<dim>::HexagonalLattice(double A_0, 
                                        double psi_0,
                                        const std::vector<dealii::Tensor<1, dim>> &dislocation_positions,
                                        const std::vector<dealii::Tensor<1, dim>> &burgers_vectors)
    : dealii::Function<dim>(3)
    , A_0(A_0)
    , psi_0(psi_0)
    , dislocation_positions(dislocation_positions)
    , burgers_vectors(burgers_vectors)
{
    q.resize(3);

    q[0][0] = 0.0;
    q[0][1] = 1.0;

    q[1][0] = 0.5 * std::sqrt(3.0);
    q[1][1] = -0.5;

    q[2][0] = -0.5 * std::sqrt(3.0);
    q[2][1] = -0.5;
}



template <int dim>
double HexagonalLattice<dim>::value(const dealii::Point<dim> &p,
                                    const unsigned int component) const
{
    double sum = 0;
    for (const auto &qn : q)
    {
        double phase_sum = 0;
        for (std::size_t i = 0; i < dislocation_positions.size(); ++i)
        {
            double s_n_j = 1.0 / (2 * M_PI) * qn * burgers_vectors[i];
            double theta_j = std::atan2(p[1] - dislocation_positions[i][1],
                                        p[0] - dislocation_positions[i][0]);
            phase_sum += s_n_j * theta_j;
        }
        sum += std::cos(qn * p - phase_sum);
    }

    if (component == 0)
        return psi_0 + A_0 * sum;
    else if (component == 1)
        return -A_0 * sum;
    else if (component == 2)
        return A_0 * sum;
    else
        throw std::invalid_argument("Phase field only has 3 components");
}

template class HexagonalLattice<2>;
template class HexagonalLattice<3>;
