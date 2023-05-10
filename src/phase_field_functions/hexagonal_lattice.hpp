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
    HexagonalLattice(double A_0, 
                     double psi_0 = -0.43, 
                     const std::vector<dealii::Tensor<1, dim>>& dislocation_positions
                     = std::vector<dealii::Tensor<1, dim>>(),
                     const std::vector<dealii::Tensor<1, dim>>& burgers_vectors 
                     = std::vector<dealii::Tensor<1, dim>>());

    virtual double value(const dealii::Point<dim> &p,
                         const unsigned int component = 0) const override;
    // virtual void vector_value(const dealii::Point<dim> &p,
    //                           dealii::Vector<double> &value) const override;
    // virtual void value_list(const std::vector<dealii::Point<dim>> &points,
    //                         std::vector<double> &values,
    //                         const unsigned int component = 0) const override;
    // virtual void vector_value_list(const std::vector<dealii::Point<dim>> &points,
    //                                std::vector<dealii::Vector<double>> &values) const override;
    
    // 4 \pi / \sqrt{3}
    static constexpr double a = 4 * M_PI * 0.5773502691896257645091487805019574556476017;

private:
    // lattice quantities
    std::vector<dealii::Tensor<1, dim>> q;
    double A_0 = 1.0;
    double psi_0 = 1.0;

    std::vector<dealii::Tensor<1, dim>> dislocation_positions;
    std::vector<dealii::Tensor<1, dim>> burgers_vectors;
};

#endif
