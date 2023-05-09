#include <iostream>
#include "parameters/toml.hpp"

void print_double(double num)
{
    std::cout << num << "\n";
}

int main(int argc, char** argv)
{
    toml::table tbl;
    try
    {
        tbl = toml::parse_file(argv[1]);
        std::cout << tbl << "\n";

        std::cout << tbl["database"]["ports"] << "\n";
        toml::array array = *tbl["database"]["ports"].as_array();

        auto vec = toml::convert<std::vector<double>>(*tbl["database"]["ports"].as_array());

        // for (const auto& item : array)
        //     std::cout << item.type() << "\n";
    }
    catch (const toml::parse_error& err)
    {
        std::cerr << "Parsing failed:\n" << err << "\n";
        return 1;
    }

    return 0;
}
