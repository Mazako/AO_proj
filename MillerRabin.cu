#include "FileLoader.h"
#include "TestRunner.h"
#include <iostream>
#include <string>
#include <cstdlib>

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: ./MillerRabin.exe <filename> <number_count> <M_CPU/S_CPU/GPU> <iterations_per_number>\n";
        return 1;
    }

    std::string filename = argv[1];
    int count = std::atoi(argv[2]);
    std::string mode = argv[3];
    int iterations_per_number = std::atoi(argv[4]);

    uint64_t* numbers = FileLoader::load_numbers_from_file(filename, count);
    if (numbers == nullptr) {
        std::cerr << "Error: Could not load numbers from file.\n";
        return 1;
    }

    TestRunner::run_tests(numbers, count, iterations_per_number, mode);

    delete[] numbers;

    return 0;
}
