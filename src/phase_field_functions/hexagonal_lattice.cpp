#include "hexagonal_lattice.hpp"

template <int dim>
HexagonalLattice<dim>::HexagonalLattice()
    : dealii::Function<dim>(3)
{}

template class HexagonalLattice<2>;
template class HexagonalLattice<3>;
