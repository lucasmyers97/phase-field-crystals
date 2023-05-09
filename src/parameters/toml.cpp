#define TOML_IMPLEMENTATION
#include "parameters/toml.hpp"

#include <vector>
#include <stdexcept>
#include <optional>

template <>
std::vector<double> toml::convert(const toml::array& array)
{
    std::vector<double> return_val;
    return_val.reserve(array.size());

    for (const auto &item : array)
    {
        std::optional<double> value = item.value<double>();
        if (!value)
            throw std::invalid_argument("Could not convert toml array to vector");

        return_val.push_back(value.value());
    }

    return return_val;
}
