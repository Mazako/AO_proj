#include "TestRunner.h"
#include "MultiThreadedMillerRabinTest.h"
#include "SingleThreadedMillerRabinTest.h"
#include "Utils.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

#include "MillerRabinExecutor.cuh"
#include "MillerRabinMultipleNumberExecutor.cuh"

void run_m_cpu_tests(uint64_t* numbers, int count, int iterations) {
    std::cout << "----------M_CPU TESTS----------\n";
    for (int i = 0; i < count; ++i) {
        MultiThreadedMillerRabinTest test(numbers[i], iterations);
        const auto [time, is_prime] = Utils::measure_time<bool>([&test]() { return test.is_prime(); });
        std::cout << "Number " << numbers[i] << (is_prime ? " PRIME" : " COMPOSITE") << "\n";
        std::cout << "Time: " << time << " ms\n";
    }
}

void run_t_cpu_tests(uint64_t* numbers, int count, int iterations) {
    std::cout << "----------T_CPU TESTS----------\n";
    for (int i = 0; i < count; ++i) {
        SingleThreadedMillerRabinTest test(numbers[i], iterations);
        const auto [time, is_prime] = Utils::measure_time<bool>([&test]() { return test.is_prime(); });
        std::cout << "Number " << numbers[i] << (is_prime ? " PRIME" : " COMPOSITE") << "\n";
        std::cout << "Time: " << time << " ms\n";
    }
}

void run_gpu_tests(uint64_t* numbers, int count, int iterations) {
    std::cout << "----------GPU TESTS----------\n";
    for (int i = 0; i < count; ++i) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        bool is_prime = miller_rabin_test_gpu(numbers[i], iterations);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        int roundedTime = static_cast<int>(std::round(elapsedTime));

        std::cout << "GPU: Number " << numbers[i] << (is_prime ? " PRIME" : " COMPOSITE") << "\n";
        std::cout << "Time: " << roundedTime << " ms\n";

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
}

void run_batch_gpu_test(uint64_t* numbers, int count, int iterations) {
    std::cout << "---------BATCH GPU TESTS--------\n";
    auto const [time, results] = Utils::measure_time<int*>([numbers, count, iterations] ()->int* {
        return miller_rabin_test_gpu_multiple(numbers, count, iterations);
    });

    for (int i = 0; i < count; i++) {
        std::cout << "GPU: Number " << numbers[i] << (results[i] == 1 ? " PRIME\n" : " COMPOSITE\n");
    }
    std::cout << "GPU Time: " << time << " ms\n";
}

void TestRunner::run_tests(uint64_t* numbers, int count, int iterations, const std::string& mode) {
    if (mode == "M_CPU") {
        run_m_cpu_tests(numbers, count, iterations);
    } else if (mode == "S_CPU") {
        run_t_cpu_tests(numbers, count, iterations);
    } else if (mode == "GPU") {
        run_gpu_tests(numbers, count, iterations);
    } else if (mode == "BATCH_GPU") {
        run_batch_gpu_test(numbers, count, iterations);
    }
    else {
        std::cerr << "Invalid mode. Use 'M_CPU', 'S_CPU', 'GPU', or 'BATCH_GPU'.\n";
    }
}
