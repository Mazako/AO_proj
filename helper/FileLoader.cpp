#include "FileLoader.h"
#include <fstream>
#include <iostream>
#include <vector>

std::pair<uint64_t*, int> FileLoader::load_numbers_from_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error: Could not open file " << filename << ".\n";
        return {nullptr, 0};
    }

    std::vector<uint64_t> numbers;
    uint64_t number;
    int loaded = 0;

    while (file >> number) {
        numbers.push_back(number);
        loaded++;
    }

    uint64_t* numbers_arr = new uint64_t[loaded];

    std::copy(numbers.begin(), numbers.end(), numbers_arr);

    return {numbers_arr, loaded};
}
