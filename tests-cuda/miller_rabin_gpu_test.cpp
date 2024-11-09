#include "../miller-cuda/MillerRabinExecutor.cuh"
#include "gtest/gtest.h"
#include <vector>

TEST(MillerRabinGPUTest, SmallPrimes) {
    std::vector<int> small_primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
    for (int prime : small_primes) {
        EXPECT_TRUE(miller_rabin_test_gpu(prime, 5)) << "Failed for prime number: " << prime;
    }
}

TEST(MillerRabinGPUTest, SmallComposites) {
    std::vector<int> small_composites = {4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20};
    for (int composite : small_composites) {
        EXPECT_FALSE(miller_rabin_test_gpu(composite, 5)) << "Failed for composite number: " << composite;
    }
}

TEST(MillerRabinGPUTest, LargePrimes) {
    std::vector<int64_t> large_primes = {
        104729,    // 10000th prime
        1299709,   // 100000th prime
        15485863   // 1000000th prime
};
    for (int64_t prime : large_primes) {
        EXPECT_TRUE(miller_rabin_test_gpu(prime, 5)) << "Failed for large prime number: " << prime;
    }
}

TEST(MillerRabinGPUTest, LargeComposites) {
    std::vector<int64_t> large_composites = {
        104730,    // 10000th prime + 1
        1299710,   // 100000th prime + 1
        15485864   // 1000000th prime + 1
};
    for (int64_t composite : large_composites) {
        EXPECT_FALSE(miller_rabin_test_gpu(composite, 5)) << "Failed for large composite number: " << composite;
    }
}

TEST(MillerRabinGPUTest, RandomLargeNumbers) {
    std::vector<int64_t> random_numbers = {
        2147483647,       // Max 32-bit int
        9223372036854775783LL, // Large prime number
        9223372036854775807LL  // Max 64-bit int
};
    for (int64_t num : random_numbers) {
        miller_rabin_test_gpu(num, 10); // Check if function runs correctly
    }
}
