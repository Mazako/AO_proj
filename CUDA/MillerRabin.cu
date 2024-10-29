#include <iostream>
#include <cuda_runtime.h>
#include "miller-rabin/MillerRabin.cu"

int main() {

    // Tablica liczb do przetestowania
    uint64_t test_numbers[] = {
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
    const int num_tests = sizeof(test_numbers) / sizeof(test_numbers[0]);

    // Uruchomienie test�w na GPU dla ka�dej liczby
    for (int i = 0; i < num_tests; ++i) {
        uint64_t number = test_numbers[i];
        const bool result = miller_rabin_test_gpu(number, iterations);
        std::cout << "GPU: Number " << number << (result ? " PRIME" : " COMPOSITE") << std::endl;
    }

    return 0;
}
