#include "FileLoader.h"
#include "TestRunner.h"
#include <iostream>
#include <string>
#include <cstdlib>

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: ./MillerRabin.exe <filename> <M_CPU/S_CPU/GPU/BATCH_GPU> <iterations_per_number>\n";
        return 1;
    }

    std::string filename = argv[1];
    std::string mode = argv[2];
    int iterations_per_number = std::atoi(argv[3]);

    auto [numbers, count] = FileLoader::load_numbers_from_file(filename);
    if (numbers == nullptr) {
        std::cerr << "Error: Could not load numbers from file.\n";
        return 1;
    }

    TestRunner::run_tests(numbers, count, iterations_per_number, mode);

    delete[] numbers;

    return 0;
}
