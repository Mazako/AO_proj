#include "MillerRabinMultipleNumberExecutor.cuh"

#include <atomic>
#include <curand_kernel.h>
#include "Utils.h"

__global__ void decompose_number_kernel(uint64_t* numbers, uint64_t* exponent_of_twos, uint64_t* odd_parts, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        auto number = numbers[idx];
        Utils::decompose_number(number - 1, exponent_of_twos[idx], odd_parts[idx]);
    }
}

__global__ void init_curand_state_kernel(curandState* states, int* results, int n, int iterations) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        results[idx] = 1;
    }

    if (idx < n * iterations) {
        curand_init(clock64(), idx, 0, &states[idx]);
    }
}

__global__ void test_kernel(uint64_t* numbers, uint64_t* exponent_of_twos, uint64_t* odd_parts, int* results, curandState* curand_states, int n, int iterations) {
    int number_idx = blockIdx.y;
    int iteration_idx = threadIdx.x + blockIdx.x * blockDim.x;


    if (number_idx >= n || iteration_idx >= iterations || results[number_idx] == 0) {
        return;
    }

    auto const base = cuda_128_specific_random(&curand_states[number_idx * iteration_idx], 2, numbers[number_idx] - 1);
    auto const odd_part = odd_parts[number_idx];
    auto const exponent_of_two = exponent_of_twos[number_idx];

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
    uint64_t* d_odd_parts;
    uint64_t* d_exponent_of_twos;
    uint64_t* d_numbers;
    curandState* d_curand_states;
    cudaStream_t stream;

    cudaMalloc(&d_numbers, arr_size);
    cudaMalloc(&d_results, results_size);
    cudaMalloc(&d_curand_states, n * iterations * sizeof(curandState));
    cudaMalloc(&d_odd_parts, arr_size);
    cudaMalloc(&d_exponent_of_twos, arr_size);

    cudaStreamCreate(&stream);

    cudaMemcpyAsync(d_numbers, numbers, results_size, cudaMemcpyHostToDevice, stream);

    init_curand_state_kernel<<< ((n * iterations) + threads_per_block - 1) / threads_per_block, threads_per_block, 0, stream>>>(d_curand_states, d_results, n, iterations);

    decompose_number_kernel<<<(n + threads_per_block - 1) / threads_per_block, threads_per_block, 0, stream>>>(d_numbers, d_exponent_of_twos, d_odd_parts, n);

    auto blockDim = dim3((iterations + threads_per_block - 1) / threads_per_block, n);
    test_kernel<<<blockDim, threads_per_block, 0, stream>>>(d_numbers, d_exponent_of_twos, d_odd_parts, d_results, d_curand_states, n, iterations);
    cudaMemcpyAsync(h_results, d_results, results_size, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);
    cudaFree(d_numbers);
    cudaFree(d_results);
    cudaFree(d_curand_states);
    cudaFree(d_numbers);
    cudaFree(d_exponent_of_twos);
    return h_results;
}
