#ifndef UTILS_H
#define UTILS_H

#include <curand_kernel.h>
#include <cstdint>

class Utils {
public:
    // Generowanie liczb losowych w CUDA
    __device__ static uint64_t get_random_number(curandState* state, uint64_t min, uint64_t max);
    __device__ static void decompose_number(uint64_t number, uint64_t& power_of_two_exponent, uint64_t& odd_component);
    __device__ static uint64_t modular_pow(uint64_t base, uint64_t exponent, uint64_t modulus);
    __device__ static bool check_composite(uint64_t candidate, uint64_t current_value, uint64_t power_of_two_exponent);
    __device__ static bool miller_rabin_test_iteration(uint64_t candidate, uint64_t power_of_two_exponent, uint64_t odd_component, uint64_t witness);
};

#endif // UTILS_H
