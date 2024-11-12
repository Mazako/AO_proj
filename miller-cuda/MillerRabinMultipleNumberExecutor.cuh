#ifndef MILLERRABINMULTIPLENUMBEREXECUTOR_CUH
#define MILLERRABINMULTIPLENUMBEREXECUTOR_CUH
#include <cstdint>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <sys/stat.h>

__device__ inline uint64_t cuda_128_specific_random(curandState* state, uint64_t min, uint64_t max) {
    return curand(state) % (max - min + 1) + min;
}

__global__ void init_curand_state_kernel(curandState* states, int* results, int n, int iterations);

__global__ void test_kernel(uint64_t* numbers, bool* results, curandState* curand_states, int n, int iterations);

int* miller_rabin_test_gpu_multiple(uint64_t* numbers, int n, int iterations, int threads_per_block = 128);

#endif //MILLERRABINMULTIPLENUMBEREXECUTOR_CUH
