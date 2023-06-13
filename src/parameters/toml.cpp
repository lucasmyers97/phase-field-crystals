#define TOML_IMPLEMENTATION
#include "parameters/toml.hpp"

#include <vector>
#include <stdexcept>
#include <optional>

#include <deal.II/base/point.h>

template <>
std::vector<double> toml::convert(const toml::array& array)
{
    if (!array.is_homogeneous(toml::node_type::floating_point))
        throw std::invalid_argument("Array does not contain only doubles in "
                                    "conversion to std::vector<double>");

    std::vector<double> return_vec;
    return_vec.reserve(array.size());
    for (const auto &item : array)
        return_vec.push_back(item.value<double>().value());

    return return_vec;
}



template <typename T>
T toml::convert(const toml::array& array)
{
    if (!array.is_homogeneous(toml::node_type::array))
        throw std::invalid_argument("Array does not contain only arrays in "
                                    "conversion to std::vector<T>");

    T return_vec;
    return_vec.reserve(array.size());
    for (const auto& item : array)
        return_vec.push_back( convert<typename T::value_type>(*item.as_array()) );

    return return_vec;
}



template
std::vector<std::vector<double>> toml::convert(const toml::array& array);

template
std::vector<std::vector<std::vector<double>>> toml::convert(const toml::array& array);


// template <>
// dealii::Point<2> toml::convert(const toml::array& array)
// {
//     dealii::Point<2> point;
// 
//     for (int i = 0; i < 2; ++i)
//     {
//         std::optional<double> value = array[i].value<double>();
//         if (!value)
//             throw std::invalid_argument("Could not convert toml array to point");
// 
//         point[i] = value.value();
//     }
// 
//     return point;
// }
// 
// 
// 
// template <>
// dealii::Point<3> toml::convert(const toml::array& array)
// {
//     dealii::Point<3> point;
// 
//     for (int i = 0; i < 3; ++i)
//     {
//         std::optional<double> value = array[i].value<double>();
//         if (!value)
//             throw std::invalid_argument("Could not convert toml array to point");
// 
//         point[i] = value.value();
//     }
// 
//     return point;
// }
// 
// 
// 
// template <>
// dealii::Tensor<1, 2> toml::convert(const toml::array& array)
// {
//     dealii::Tensor<1, 2> point;
// 
//     for (int i = 0; i < 2; ++i)
//     {
//         std::optional<double> value = array[i].value<double>();
//         if (!value)
//             throw std::invalid_argument("Could not convert toml array to point");
// 
//         point[i] = value.value();
//     }
// 
//     return point;
// }
// 
// 
// 
// template <>
// dealii::Tensor<1, 3> toml::convert(const toml::array& array)
// {
//     dealii::Tensor<1, 3> point;
// 
//     for (int i = 0; i < 3; ++i)
//     {
//         std::optional<double> value = array[i].value<double>();
//         if (!value)
//             throw std::invalid_argument("Could not convert toml array to point");
// 
//         point[i] = value.value();
//     }
// 
//     return point;
// }
// 
// 
// 
// template <>
// std::vector<dealii::Tensor<1, 2>> toml::convert(const toml::array& array)
// {
//     std::vector<dealii::Tensor<1, 2>> point_list;
//     point_list.reserve(array.size());
// 
//     for (const auto& item : array)
//     {
//         if (const toml::array* in_array = item.as_array())
//         {
//             point_list.push_back( toml::convert<dealii::Tensor<1, 2>>(*in_array) );
//         }
//         else
//         {
//             throw std::invalid_argument("Cannot convert array to vector of tensors");
//         }
//     }
// 
//     return point_list;
// }
// 
// 
// 
// template <>
// std::vector<dealii::Tensor<1, 3>> toml::convert(const toml::array& array)
// {
//     std::vector<dealii::Tensor<1, 3>> point_list;
//     point_list.reserve(array.size());
// 
//     for (const auto& item : array)
//     {
//         if (const toml::array* in_array = item.as_array())
//         {
//             point_list.push_back( toml::convert<dealii::Tensor<1, 3>>(*in_array) );
//         }
//         else
//         {
//             throw std::invalid_argument("Cannot convert array to vector of tensors");
//         }
//     }
// 
//     return point_list;
// }
