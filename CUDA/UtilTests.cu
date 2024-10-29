#include <iostream>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include "helper/Utils.cu"

// Kernel do testowania generowania liczb losowych
__global__ void test_random_number_kernel(uint64_t* results, int num_samples, uint64_t min, uint64_t max, unsigned long long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_samples) {
        // Inicjalizuj stan curand
        curandState state;
        curand_init(seed, idx, 0, &state);

        // U¿yj funkcji do generowania losowej liczby
        results[idx] = Utils::get_random_number(&state, min, max);
    }
}

void test_random_number(int num_samples, uint64_t min, uint64_t max) {
    // Alokacja pamiêci
    uint64_t* d_results;
    cudaMalloc(&d_results, num_samples * sizeof(uint64_t));

    // Wykonanie j¹dra
    int threads_per_block = 256;
    int blocks = (num_samples + threads_per_block - 1) / threads_per_block;
    test_random_number_kernel << <blocks, threads_per_block >> > (d_results, num_samples, min, max, time(NULL));

    // Kopiowanie wyników z powrotem na CPU
    uint64_t* h_results = new uint64_t[num_samples];
    cudaMemcpy(h_results, d_results, num_samples * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Wyœwietlanie wyników
    std::cout << "Generated random numbers:\n";
    for (int i = 0; i < num_samples; ++i) {
        std::cout << "Random number " << i << ": " << h_results[i] << "\n";
    }

    delete[] h_results;
    cudaFree(d_results);
}

int main() {

    int num_samples = 20; 
    uint64_t min = 10;      
    uint64_t max = 100;     

    test_random_number(num_samples, min, max);

    return 0;
}