#include "hexagonal_lattice.hpp"

#include <deal.II/base/point.h>

#include <cmath>

template <int dim>
HexagonalLattice<dim>::HexagonalLattice()
    : dealii::Function<dim>(3)
{
    q.resize(3);

    q[0][0] = 0.0;
    q[0][1] = 1.0;

    q[1][0] = std::sqrt(3.0) / 2.0;
    q[1][1] = -0.5;

    q[2][0] = -std::sqrt(3.0) / 2.0;
    q[2][1] = -0.5;
}



template <int dim>
HexagonalLattice<dim>::HexagonalLattice(double A_0, double psi_0)
    : dealii::Function<dim>(3)
    , A_0(A_0)
    , psi_0(psi_0)
{
    q.resize(3);

    q[0][0] = 0.0;
    q[0][1] = 1.0;

    q[1][0] = std::sqrt(3.0) / 2.0;
    q[1][1] = -0.5;

    q[2][0] = -std::sqrt(3.0) / 2.0;
    q[2][1] = -0.5;
}

template <int dim>
double HexagonalLattice<dim>::value(const dealii::Point<dim> &p,
                                    const unsigned int component) const
{
    const int sign = ((component % 2) == 0) ? 1 : -1;
    const double local_psi_0 = (component == 0) ? psi_0 : 0;

    double sum = 0;
    for (const auto &qn : q)
        sum += std::cos(qn * p);

    return local_psi_0 + A_0 * sign * sum;
}

template class HexagonalLattice<2>;
template class HexagonalLattice<3>;
