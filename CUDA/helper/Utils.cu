#include <curand_kernel.h>
#include <cstdint>

class Utils {
public:
    
    __device__ static uint64_t get_random_number(curandState* state, uint64_t min, uint64_t max) {
        return curand(state) % (max - min + 1) + min;
    }

    __device__ static void decompose_number(uint64_t number, uint64_t& power_of_two_exponent, uint64_t& odd_component) {
        power_of_two_exponent = 0;
        odd_component = number;
        while ((odd_component & 1) == 0) {
            odd_component >>= 1;
            ++power_of_two_exponent;
        }
    }

    __device__ static uint64_t modular_pow(uint64_t base, uint64_t exponent, uint64_t modulus) {
        uint64_t result = 1;
        base %= modulus;

        while (exponent > 0) {
            if (exponent % 2 == 1) {
                result = (result * base) % modulus;
            }
            base = (base * base) % modulus;  
            exponent /= 2;
        }
        return result;
    }

    __device__ static bool check_composite(uint64_t candidate, uint64_t current_value, uint64_t power_of_two_exponent) {
        for (uint64_t round = 1; round < power_of_two_exponent; ++round) {
            current_value = modular_pow(current_value, 2, candidate);

            if (current_value == candidate - 1)
                return false;

            if (current_value == 1)
                return true;
        }
        return true;
    }

    __device__ static bool miller_rabin_test_iteration(uint64_t candidate, uint64_t power_of_two_exponent, uint64_t odd_component, uint64_t witness) {
        uint64_t current_value = modular_pow(witness, odd_component, candidate);

        if (current_value == 1 || current_value == candidate - 1)
            return true;

        return !check_composite(candidate, current_value, power_of_two_exponent);
    }
};