#ifndef CUDARNGWARMUP_CUH
#define CUDARNGWARMUP_CUH

#include <cstdint>
#include <cuda_runtime.h>

__global__ void warmupTestRandomNumberKernel(uint64_t* results, int num_samples, uint64_t min, uint64_t max, unsigned long long seed);

void warmupTestRandomWarmupPerformer(int num_samples, uint64_t min, uint64_t max);

#endif //CUDARNGWARMUP_CUH
