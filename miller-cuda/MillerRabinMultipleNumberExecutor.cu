#include "MillerRabinMultipleNumberExecutor.cuh"

#include <atomic>
#include <cstdio>
#include <curand_kernel.h>
#include <sys/stat.h>
#include "Utils.h"

__global__ void init_curand_state_kernel(curandState* states, int* results, int n, int iterations) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        results[idx] = 1;
    }

    if (idx < n * iterations) {
        curand_init(clock64(), idx, 0, &states[idx]);
    }
}

__global__ void test_kernel(uint64_t* numbers, int* results, curandState* curand_states, int n, int iterations) {
    int number_idx = blockIdx.y;
    int iteration_idx = threadIdx.x + blockIdx.x * blockDim.x;


    if (number_idx >= n || iteration_idx >= iterations || results[number_idx] == 0) {
        return;
    }

    auto const base = cuda_128_specific_random(&curand_states[number_idx * iteration_idx], 2, numbers[number_idx] - 1);
    uint64_t exponent_of_two, odd_part;
    Utils::decompose_number(numbers[number_idx] - 1, exponent_of_two, odd_part);

    uint64_t x = Utils::modular_pow(base, odd_part, numbers[number_idx]);

    if (x == 1 || x == numbers[number_idx] - 1) {
        atomicOr(&results[number_idx], 1);
        return;
    }

    for (uint64_t j = 0; j < exponent_of_two - 1; j++) {
        x = Utils::modular_pow(x, 2, numbers[number_idx]);
        if (x == numbers[number_idx] - 1) {
            atomicOr(&results[number_idx], 1);
            return;
        }
    }

    atomicAnd(&results[number_idx], 0);
}

int* miller_rabin_test_gpu_multiple(uint64_t* numbers, int n, int iterations, int threads_per_block) {
    unsigned long arr_size = n * sizeof(uint64_t);
    int results_size = n * sizeof(int);

    int* h_results = (int*)malloc(results_size);

    int* d_results;
    uint64_t* d_numbers;
    curandState* d_curand_states;

    cudaMalloc(&d_numbers, arr_size);
    cudaMalloc(&d_results, results_size);
    cudaMalloc(&d_curand_states, n * iterations * sizeof(curandState));

    cudaMemcpy(d_numbers, numbers, results_size, cudaMemcpyHostToDevice);

    init_curand_state_kernel<<< ((n * iterations) + threads_per_block - 1) / threads_per_block, threads_per_block>>>(d_curand_states, d_results, n, iterations);
    cudaDeviceSynchronize();

    auto blockDim = dim3((iterations + threads_per_block - 1) / threads_per_block, n);
    test_kernel <<<blockDim, threads_per_block>>>(d_numbers, d_results, d_curand_states, n, iterations);
    cudaDeviceSynchronize();

    cudaMemcpy(h_results, d_results, results_size, cudaMemcpyDeviceToHost);

    cudaFree(d_numbers);
    cudaFree(d_results);
    cudaFree(d_curand_states);
    return h_results;
}