#include "FileLoader.h"
#include "TestRunner.h"
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: ./MillerRabin.exe <filename> <number_count> <M_CPU/T_CPU/GPU>\n";
        return 1;
    }

    std::string filename = argv[1];
    int count = std::atoi(argv[2]);
    std::string mode = argv[3];
    int iterations = 100000;

    std::vector<uint64_t> numbers = FileLoader::load_numbers_from_file(filename, count);
    if (numbers.empty()) {
        std::cerr << "Error: Could not load numbers from file.\n";
        return 1;
    }
    TestRunner::run_tests(numbers, iterations, mode);

    return 0;
}
