#include "FileLoader.h"
#include <fstream>
#include <iostream>

std::vector<uint64_t> FileLoader::load_numbers_from_file(const std::string& filename, int count) {
    std::ifstream file(filename);
    std::vector<uint64_t> numbers;
    uint64_t number;
    int loaded = 0;

    while (file >> number && loaded < count) {
        numbers.push_back(number);
        loaded++;
    }

    if (numbers.size() != count) {
        std::cerr << "Warning: Expected " << count << " numbers, but found " << numbers.size() << ".\n";
    }

    return numbers;
}
