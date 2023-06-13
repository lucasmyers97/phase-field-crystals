#ifndef TOML_HPP
#define TOML_HPP

#define TOML_HEADER_ONLY 0
#include <tomlplusplus/toml.hpp>

#include <vector>

#include <deal.II/base/point.h>

namespace toml
{
    template <typename T>
    T convert(const array&);

    // template <>
    // std::vector<double> convert(const array&);

    // template <>
    // dealii::Tensor<1, 2> convert(const array&);

    // template <>
    // dealii::Tensor<1, 3> convert(const array&);

    // template <>
    // std::vector<dealii::Tensor<1, 2>> convert(const array&);

    // template <>
    // std::vector<dealii::Tensor<1, 3>> convert(const array&);

    // template <>
    // dealii::Point<2> convert(const array&);

    // template <>
    // dealii::Point<3> convert(const array&);
} // toml

#endif
