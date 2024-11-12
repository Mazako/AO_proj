#include "FileLoader.h"
#include <fstream>
#include <iostream>

uint64_t* FileLoader::load_numbers_from_file(const std::string& filename, int count) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error: Could not open file " << filename << ".\n";
        return nullptr;
    }

    uint64_t* numbers = new uint64_t[count];
    uint64_t number;
    int loaded = 0;

    while (file >> number && loaded < count) {
        numbers[loaded] = number;
        loaded++;
    }

    if (loaded != count) {
        std::cerr << "Warning: Expected " << count << " numbers, but found " << loaded << ".\n";
    }
    return numbers;
}
