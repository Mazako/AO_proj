#include <cstdint>
#include <iostream>
#include <vector>

#include "MillerRabinExecutor.cuh"
#include "SingleThreadedMillerRabinTest.h"
#include "Utils.h"
#include "CudaRngWarmup.cuh"

void run_tests(const std::vector<uint64_t>& numbers, int iterations) {
    std::cout << "----------CPU TESTS----------" << std::endl;
    for (auto number : numbers) {
        SingleThreadedMillerRabinTest test(number, iterations);
        const auto [time, value] = Utils::measure_time<bool>([test]()-> bool {return test.is_prime();});
        std::cout << "Number " << number << (value ? " PRIME" : " COMPOSITE") << std::endl;
        std::cout << "Time: " << time << " ns\n";
    }

    std::cout << "----------GPU TESTS----------" << std::endl;

    for (unsigned long long number : numbers) {
        const bool result = miller_rabin_test_gpu(number, iterations);
        std::cout << "GPU: Number " << number << (result ? " PRIME" : " COMPOSITE") << std::endl;
    }
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
            18446744073709551557,   // prime (może być zbyt duża)
            2305843009213693951,    // prime
            179424673,              // prime
            32452843,               // prime
            15485867,               // prime
            10000000019             // prime, ale pokazuje composite?
    };


    std::cout << "----------GPU RNG generator tests----------" << std::endl;
    warmupTestRandomWarmupPerformer(10, 1, 1000000);

    const int iterations = 100000;
    run_tests(test_numbers, iterations);

    return 0;
}
