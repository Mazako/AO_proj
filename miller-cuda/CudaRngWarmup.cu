#include "CudaRngWarmup.cuh"
#include <curand_kernel.h>
#include <iostream>

#include "MillerRabinExecutor.cuh"

__global__ void warmupTestRandomNumberKernel(uint64_t *results, int num_samples, uint64_t min, uint64_t max,
                                             unsigned long long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_samples) {
        // Inicjalizuj stan curand
        curandState state;
        curand_init(seed, idx, 0, &state);

        // U�yj funkcji do generowania losowej liczby
        results[idx] = cudaSpecificRandom(&state, min, max);
    }
}

void warmupTestRandomWarmupPerformer(int num_samples, uint64_t min, uint64_t max) {
    // Alokacja pami�ci
    uint64_t *d_results;
    cudaMalloc(&d_results, num_samples * sizeof(uint64_t));

    // Wykonanie j�dra
    int threads_per_block = 256;
    int blocks = (num_samples + threads_per_block - 1) / threads_per_block;
    warmupTestRandomNumberKernel << <blocks, threads_per_block >> >(d_results, num_samples, min, max, time(NULL));

    // Kopiowanie wynik�w z powrotem na CPU
    uint64_t *h_results = new uint64_t[num_samples];
    cudaMemcpy(h_results, d_results, num_samples * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Wy�wietlanie wynik�w
    std::cout << "Generated random numbers:\n";
    for (int i = 0; i < num_samples; ++i) {
        std::cout << "Random number " << i << ": " << h_results[i] << "\n";
    }

    delete[] h_results;
    cudaFree(d_results);
}
