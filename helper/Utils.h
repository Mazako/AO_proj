#ifndef UTILS_H
#define UTILS_H

#include <chrono>
#include <cstdint>
#include <functional>

class Utils {
public:
    static uint64_t modular_pow(uint64_t base, uint64_t exponent, uint64_t modulus);
    static void decompose_number(uint64_t number, uint64_t& power_of_two_exponent, uint64_t& odd_component);
    static uint64_t get_random_number(uint64_t min, uint64_t max);
    static bool check_composite(uint64_t candidate, uint64_t current_value, uint64_t power_of_two_exponent);
    static bool miller_rabin_test_iteration(uint64_t candidate, uint64_t power_of_two_exponent, uint64_t odd_component, uint64_t witness);

    template<typename T>
    static std::pair<long long, T> measure_time(const std::function<T()>& callable) {
        const std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        T result = callable();
        const std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        long long time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        return {time, result};
    }
};

#endif // UTILS_H
