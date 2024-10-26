#include <cstdint>
#include <iostream>
#include <vector>
#include "miller-rabin/SingleThreadedMillerRabinTest.h"

void run_tests(const std::vector<uint64_t>& numbers, int iterations) {
    for (auto number : numbers) {
        SingleThreadedMillerRabinTest test(number, iterations);
        const bool result = test.is_prime();

        std::cout << "Number " << number << (result ? " PRIME" : " COMPOSITE") << std::endl;
    }
}

int main() {
    std::vector<uint64_t> test_numbers = {
        100000007,              // prime
        100000037,              // prime
        100000039,              // prime
        100000041,              // composite
        123456789,              // composite
        4294967311,             // prime
        67280421310721,         // prime
        999999000000000003,     // composite
        999999999989,           // prime
        67280421310721,         // prime
        18446744073709551557,   // prime
        2305843009213693951     // prime
    };

    const int iterations = 100000;
    run_tests(test_numbers, iterations);

    return 0;
}
