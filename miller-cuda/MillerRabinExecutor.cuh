#ifndef MILLERRABINEXECUTOR_CUH
#define MILLERRABINEXECUTOR_CUH
#include <cuda_runtime.h>
#include <curand_kernel.h>

__device__ uint64_t cudaSpecificRandom(curandState* state, uint64_t min, uint64_t max);

__global__ void miller_rabin_kernel(uint64_t* number, int iterations, bool* result, curandState* states);

bool miller_rabin_test_gpu(uint64_t number, int iterations);


#endif //MILLERRABINEXECUTOR_CUH
