#ifndef HEXAGONAL_LATTICE_HPP
#define HEXAGONAL_LATTICE_HPP

#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>

#include <vector>

template <int dim>
class HexagonalLattice : public dealii::Function<dim>
{
public:
    HexagonalLattice();

private:
    std::vector<dealii::Tensor<1, dim>> q;
    double A_0;
    double psi_0;
};

#endif
