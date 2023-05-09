#ifndef TOML_HPP
#define TOML_HPP

#define TOML_HEADER_ONLY 0
#include <tomlplusplus/toml.hpp>

#include <vector>

namespace toml
{
    template <typename T>
    T convert(const array&);

    template <>
    std::vector<double> convert(const array&);
} // toml

#endif
