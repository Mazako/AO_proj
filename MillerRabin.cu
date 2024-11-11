#include <cstdint>
#include <iostream>
#include <vector>

#include "MillerRabinExecutor.cuh"
#include "SingleThreadedMillerRabinTest.h"
#include "Utils.h"
#include "CudaRngWarmup.cuh"
#include "MillerRabinMultipleNumberExecutor.cuh"

void run_tests(const std::vector<uint64_t>& numbers, int iterations) {
    std::cout << "----------CPU TESTS----------" << std::endl;
    for (auto number : numbers) {
        SingleThreadedMillerRabinTest test(number, iterations);
        const auto [time, value] = Utils::measure_time<bool>([test]()-> bool {return test.is_prime();});
        std::cout << "Number " << number << (value ? " PRIME" : " COMPOSITE") << std::endl;
        std::cout << "Time: " << time << " ms\n";
    }

    std::cout << "----------GPU TESTS----------" << std::endl;
    bool* results = new bool[numbers.size()];

    const auto [time, _] = Utils::measure_time<void*>([numbers, iterations, results] ()->void* {
        void* r = nullptr;
        for (int i = 0; i < numbers.size(); i++) {
            const bool result = miller_rabin_test_gpu(numbers[i], iterations);
            results[i] = result;
        }
        return r;

    });

    std::cout << "TIME: " << time << " ms\n";
    for (int i = 0; i < numbers.size(); i++) {
        std::cout << "GPU: Number " << numbers[i] << (results[i] ? " PRIME" : " COMPOSITE") << std::endl;
    }

    delete results;
}

void run_multi_gpu_test(const std::vector<uint64_t>& numbers, int iterations) {
    uint64_t* nums_ptr = new uint64_t[numbers.size()];

    for (int i = 0; i < numbers.size(); i++) {
        nums_ptr[i] = numbers[i];
    }

    const auto [time, results] = Utils::measure_time<int*>([nums_ptr, numbers, iterations]()->int* {return miller_rabin_test_gpu_multiple(nums_ptr, numbers.size(), iterations);});

    std::cout << "TIME: " << time << " ms\n";

    for (int i = 0; i < numbers.size(); i++) {
        std::cout << nums_ptr[i] << ": " << (results[i] ? "PRIME" : "COMPOSITE") << std::endl;
    }

    delete nums_ptr;
}

int main() {
    std::vector<uint64_t> test_numbers = {
            100000007,              // prime
            100000037,              // prime
            100000039,              // prime
            100000041,              // composite
            123456789,              // composite
            4294967311,             // prime
            67280421310721,         // prime
            999999000000000003,     // composite
            999999999989,           // prime
            67280421310721,         // prime
            18446744073709551557,   // prime (biggest 64bit prime)
            2305843009213693951,    // prime
            179424673,              // prime
            32452843,               // prime
            15485867,               // prime
            10000000019,            // prime
    };

    std::cout << "----------GPU RNG generator tests----------" << std::endl;
    warmupTestRandomWarmupPerformer(10, 1, 1000000);

    const int iterations = 100000;
    run_tests(test_numbers, iterations);

    std::cout << "------------MultiGpu tests-------------\n";

    run_multi_gpu_test(test_numbers, iterations);
    return 0;
}
