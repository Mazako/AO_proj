#include "Utils.h"

#include <chrono>
#include <random>

uint64_t Utils::modular_pow(uint64_t base, uint64_t exponent, uint64_t modulus) {
    uint64_t result = 1;
    base %= modulus;

    while (exponent > 0) {
        if (exponent % 2 == 1) {
            result = static_cast<__uint128_t>(result) * base % modulus;
        }
        base = static_cast<__uint128_t>(base) * base % modulus;
        exponent /= 2;
    }
    return result;
}

void Utils::decompose_number(uint64_t number, uint64_t& power_of_two_exponent, uint64_t& odd_component) {
    power_of_two_exponent = 0;
    odd_component = number;
    while ((odd_component & 1) == 0) {
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

bool Utils::check_composite(uint64_t candidate, uint64_t current_value, uint64_t power_of_two_exponent) {
    for (uint64_t round = 1; round < power_of_two_exponent; ++round) {
        current_value = modular_pow(current_value, 2, candidate);

        if (current_value == candidate - 1)
            return false;

        if (current_value == 1)
            return true;
    }
    return true;
}

bool Utils::miller_rabin_test_iteration(uint64_t candidate, uint64_t power_of_two_exponent, uint64_t odd_component, uint64_t witness) {
    uint64_t current_value = modular_pow(witness, odd_component, candidate);

    if (current_value == 1 || current_value == candidate - 1)
        return true;

    return !check_composite(candidate, current_value, power_of_two_exponent);
}
