#ifndef HEXAGONAL_LATTICE_HPP
#define HEXAGONAL_LATTICE_HPP

#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/point.h>

#include <deal.II/lac/vector.h>

#include <vector>

template <int dim>
class HexagonalLattice : public dealii::Function<dim>
{
public:
    HexagonalLattice();

    virtual double value(const dealii::Point<dim> &p,
                         const unsigned int component = 0) const override;
    // virtual void vector_value(const dealii::Point<dim> &p,
    //                           dealii::Vector<double> &value) const override;
    // virtual void value_list(const std::vector<dealii::Point<dim>> &points,
    //                         std::vector<double> &values,
    //                         const unsigned int component = 0) const override;
    // virtual void vector_value_list(const std::vector<dealii::Point<dim>> &points,
    //                                std::vector<dealii::Vector<double>> &values) const override;

private:
    std::vector<dealii::Tensor<1, dim>> q;
    double A_0 = 1.0;
    double psi_0 = 1.0;
};

#endif
