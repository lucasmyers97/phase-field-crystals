#include "vector_conversion.hpp"

namespace vector_conversion
{

template <>
dealii::Point<1> convert(const std::vector<double>& vec)
{
    if (vec.size() != 1)
        throw std::invalid_argument("std::vector size does not match dimension "
                                    "in conversion to dealii::Point");

    dealii::Point<1> p(vec[0]);
    return p;
}



template <>
dealii::Point<2> convert(const std::vector<double>& vec)
{
    if (vec.size() != 2)
        throw std::invalid_argument("std::vector size does not match dimension "
                                    "in conversion to dealii::Point");

    dealii::Point<2> p(vec[0], vec[1]);
    return p;
}



template <>
dealii::Point<3> convert(const std::vector<double>& vec)
{
    if (vec.size() != 3)
        throw std::invalid_argument("std::vector size does not match dimension "
                                    "in conversion to dealii::Point");

    dealii::Point<3> p(vec[0], vec[1], vec[2]);
    return p;
}

} // vector_convesion
