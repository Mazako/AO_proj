#include "Utils.h"
#include "MillerRabinNsightExecutor.cuh"

__device__ uint64_t cudaSpecificRandomNsight(curandState* state, uint64_t min, uint64_t max) {

    // Zwrócenie min gdy zakres jest niewłaściwy
    if (min > max) {
        return min;
    }
    return curand(state) % (max - min + 1) + min;
}

__global__ void miller_rabin_nsight_kernel(uint64_t* number, int iterations, bool* result, curandState* states) {
    // Pamięć współdzielona
    __shared__ bool shared_results[256];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= iterations) return;

    // Wywołanie curand_init raz na wątek zamiast dla każdego wątku
    //curandState localState;
    //curand_init(clock64(), idx, 0, &localState);

    //const uint64_t base = cudaSpecificRandomNsight(&states[idx], 2, *number - 2);
    //uint64_t exponent_of_two, odd_part;
    //Utils::decompose_number(*number - 1, exponent_of_two, odd_part);

    // Inicjalizacja stanu generatora liczb losowych
    curandState localState = states[idx];

    const uint64_t base = cudaSpecificRandomNsight(&localState, 2, *number - 2);
    uint64_t exponent_of_two, odd_part;
    Utils::decompose_number(*number - 1, exponent_of_two, odd_part);

    uint64_t x = Utils::modular_pow(base, odd_part, *number);

    if (x == 1 || x == *number - 1) {
        shared_results[threadIdx.x] = true;
        return;
    }

    for (uint64_t j = 0; j < exponent_of_two - 1; ++j) {
        x = Utils::modular_pow(x, 2, *number);
        if (x == *number - 1) {
            shared_results[threadIdx.x] = true;
            return;
        }
    }

    shared_results[threadIdx.x] = false;

    // Synchronizacja wątków w bloku przed zbieraniem wyników
    __syncthreads();

    // Agregowanie wyników w obrębie bloku
    if (threadIdx.x == 0) {
        bool all_true = true;
        for (int i = 0; i < blockDim.x; ++i) {
            if (!shared_results[i]) {
                all_true = false;
                break;
            }
        }
        result[blockIdx.x] = all_true;
    }
}

bool miller_rabin_test_gpu_nsight(uint64_t number, int iterations) {

    // Szybka obsługa małych liczb
    if (number <= 3) {
        return number > 1;
    }

    uint64_t* d_number;
    bool* d_results, * h_results;
    h_results = new bool[iterations];

    // Alokacja pami�ci na GPU
    cudaMalloc((void**)&d_number, sizeof(uint64_t));
    cudaMalloc((void**)&d_results, iterations * sizeof(bool));
    curandState* d_states;
    cudaMalloc(&d_states, iterations * sizeof(curandState));

    // Przekazywanie liczby do GPU
    cudaMemcpy(d_number, &number, sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Ustalanie liczby blok�w i w�tk�w
    int threads_per_block = 256;
    int blocks_per_grid = (iterations + threads_per_block - 1) / threads_per_block;
    miller_rabin_nsight_kernel << <blocks_per_grid, threads_per_block >> > (d_number, iterations, d_results, d_states);

    // Kopiowanie wynik�w z GPU
    cudaMemcpy(h_results, d_results, iterations * sizeof(bool), cudaMemcpyDeviceToHost);

    bool is_prime = true;
    for (int i = 0; i < iterations; ++i) {
        if (!h_results[i]) {
            is_prime = false;
            break;
        }
    }

    // Zwolnienie pami�ci
    cudaFree(d_number);
    cudaFree(d_results);
    cudaFree(d_states);
    delete[] h_results;

    return is_prime;
}