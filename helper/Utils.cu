#include "Utils.h"
#include <random>


__host__ __device__ uint64_t Utils::overflow_save_mod_mul(uint64_t a, uint64_t b, uint64_t m) {
    uint64_t res = 0;
    uint64_t temp_b;

    if (b >= m) {
        if (m > UINT64_MAX / 2u)
            b -= m;
        else
            b %= m;
    }

    while (a != 0) {
        if (a & 1) {
            if (b >= m - res)
                res -= m;
            res += b;
        }
        a >>= 1;

        temp_b = b;
        if (b >= m - b)
            temp_b -= m;
        b += temp_b;
    }
    return res;
}

__host__ __device__ uint64_t Utils::mod_mul(uint64_t a, uint64_t b, uint64_t m) {
    if (a > UINT64_MAX / b) {
        return overflow_save_mod_mul(a, b, m);
    }
    return a * b % m;
}

__host__ __device__ uint64_t Utils::modular_pow(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) {
            result = mod_mul(result, base, mod);
        }
        base = mod_mul(base, base, mod);
        exp >>= 1;
    }
    return result;
}

__host__ __device__ void Utils::decompose_number(uint64_t number, uint64_t &power_of_two_exponent,
                                                 uint64_t &odd_component) {
    power_of_two_exponent = 0;
    odd_component = number;
    while ((odd_component & 1) == 0) {    // only last bit of odd_component is being compared to 1 in AND logic operation
        odd_component >>= 1;
        ++power_of_two_exponent;
    }
}

uint64_t Utils::get_random_number(uint64_t min, uint64_t max) {
    std::random_device random_device;
    std::mt19937_64 generator(random_device());
    std::uniform_int_distribution<uint64_t> distribution(min, max);
    return distribution(generator);
}

__host__ __device__ bool Utils::check_composite(uint64_t candidate, uint64_t current_value,
                                                uint64_t power_of_two_exponent) {
    for (uint64_t round = 1; round < power_of_two_exponent; ++round) {
        current_value = modular_pow(current_value, 2, candidate);

        if (current_value == candidate - 1)
            return false;

        if (current_value == 1)
            return true;
    }
    return true;
}

__host__ __device__ bool Utils::miller_rabin_test_iteration(uint64_t candidate, uint64_t power_of_two_exponent,
                                                            uint64_t odd_component, uint64_t witness) {
    uint64_t current_value = modular_pow(witness, odd_component, candidate);

    if (current_value == 1 || current_value == candidate - 1)
        return true;

    return !check_composite(candidate, current_value, power_of_two_exponent);
}
