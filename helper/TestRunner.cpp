#include "TestRunner.h"
#include "MultiThreadedMillerRabinTest.h"
#include "SingleThreadedMillerRabinTest.h"
#include "Utils.h"
#include <iostream>
#include <future>
#include <vector>
#include <cuda_runtime.h>

#include "MillerRabinExecutor.cuh"

void run_m_cpu_tests(const std::vector<uint64_t>& numbers, int iterations) {
    std::cout << "----------M_CPU TESTS----------\n";
    for (auto number : numbers) {
        MultiThreadedMillerRabinTest test(number, iterations);
        const auto [time, is_prime] = Utils::measure_time<bool>([&test]() { return test.is_prime(); });
        std::cout << "Number " << number << (is_prime ? " PRIME" : " COMPOSITE") << "\n";
        std::cout << "Time: " << time << " ms\n";
    }
}

void run_t_cpu_tests(const std::vector<uint64_t>& numbers, int iterations) {
    std::cout << "----------T_CPU TESTS----------\n";
    for (auto number : numbers) {
        SingleThreadedMillerRabinTest test(number, iterations);
        const auto [time, is_prime] = Utils::measure_time<bool>([&test]() { return test.is_prime(); });
        std::cout << "Number " << number << (is_prime ? " PRIME" : " COMPOSITE") << "\n";
        std::cout << "Time: " << time << " ms\n";
    }
}

void run_gpu_tests(const std::vector<uint64_t>& numbers, int iterations) {
    std::cout << "----------GPU TESTS----------\n";
    for (auto number : numbers) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        bool is_prime = miller_rabin_test_gpu(number, iterations);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        int roundedTime = static_cast<int>(std::round(elapsedTime));

        std::cout << "GPU: Number " << number << (is_prime ? " PRIME" : " COMPOSITE") << "\n";
        std::cout << "Time: " << roundedTime << " ms\n";

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
}

void TestRunner::run_tests(const std::vector<uint64_t>& numbers, int iterations, const std::string& mode) {
    if (mode == "M_CPU") {
        run_m_cpu_tests(numbers, iterations);
    } else if (mode == "T_CPU") {
        run_t_cpu_tests(numbers, iterations);
    } else if (mode == "GPU") {
        run_gpu_tests(numbers, iterations);
    } else {
        std::cerr << "Invalid mode. Use 'M_CPU', 'T_CPU', or 'GPU'.\n";
    }
}
