#ifndef MILLERRABINENSIGHTXECUTOR_CUH
#define MILLERRABINENSIGHTXECUTOR_CUH
#include <curand_kernel.h>
#include <cstdint>

__device__ uint64_t cudaSpecificRandomNsight(curandState* state, uint64_t min, uint64_t max);

__global__ void miller_rabin_nsight_kernel(uint64_t* number, int iterations, bool* result, curandState* states);

bool miller_rabin_test_gpu_nsight(uint64_t number, int iterations);

#endif //MILLERRABINENSIGHTXECUTOR_CUH
