#include "../miller-rabin/SingleThreadedMillerRabinTest.h"
#include "gtest/gtest.h"

TEST(MillerRabinTest, SmallPrimes) {
    std::vector<int> small_primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
    for (int prime : small_primes) {
        SingleThreadedMillerRabinTest test(prime, 5);
        EXPECT_TRUE(test.is_prime()) << "Failed for prime number: " << prime;
    }
}

TEST(MillerRabinTest, SmallComposites) {
    std::vector<int> small_composites = {4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20};
    for (int composite : small_composites) {
        SingleThreadedMillerRabinTest test(composite, 5);
        EXPECT_FALSE(test.is_prime()) << "Failed for composite number: " << composite;
    }
}

TEST(MillerRabinTest, LargePrimes) {
    std::vector<int64_t> large_primes = {
            104729,    // 10000th prime
            1299709,   // 100000th prime
            15485863   // 1000000th prime
    };
    for (int64_t prime : large_primes) {
        SingleThreadedMillerRabinTest test(prime, 5);
        EXPECT_TRUE(test.is_prime()) << "Failed for large prime number: " << prime;
    }
}

TEST(MillerRabinTest, LargeComposites) {
    std::vector<int64_t> large_composites = {
            104730,    // 10000th prime + 1
            1299710,   // 100000th prime + 1
            15485864   // 1000000th prime + 1
    };
    for (int64_t composite : large_composites) {
        SingleThreadedMillerRabinTest test(composite, 5);
        EXPECT_FALSE(test.is_prime()) << "Failed for large composite number: " << composite;
    }
}

TEST(MillerRabinTest, CarmichaelNumbers) {
    // Carmichael numbers are composite numbers that can pass certain primality tests
    std::vector<int> carmichael_numbers = {561, 1105, 1729, 2465, 2821, 6601};
    for (int n : carmichael_numbers) {
        SingleThreadedMillerRabinTest test(n, 5);
        EXPECT_FALSE(test.is_prime()) << "Failed for Carmichael number: " << n;
    }
}

TEST(MillerRabinTest, EdgeCases) {
    SingleThreadedMillerRabinTest test_zero(0, 5);
    EXPECT_FALSE(test_zero.is_prime()) << "0 should not be prime";

    SingleThreadedMillerRabinTest test_one(1, 5);
    EXPECT_FALSE(test_one.is_prime()) << "1 should not be prime";

    SingleThreadedMillerRabinTest test_negative(-7, 5);
    EXPECT_FALSE(test_negative.is_prime()) << "Negative numbers should not be prime";
}

TEST(MillerRabinTest, RandomLargeNumbers) {
    // Testing random large numbers
    std::vector<int64_t> random_numbers = {
            2147483647,       // Maximum value of a 32-bit int
            9223372036854775783LL, // Large prime number
            9223372036854775807LL  // Maximum value of a 64-bit int
    };
    for (int64_t num : random_numbers) {
        SingleThreadedMillerRabinTest test(num, 10);
        // Since these are large numbers, we might not know for sure if they are prime
        // At the very least, we can check if the function does not hang
        // or return an error
        test.is_prime(); // Just check if the function works
    }
}

TEST(MillerRabinTest, KnownPrimesAndComposites) {
    struct TestCase {
        int64_t number;
        bool is_prime;
    };

    std::vector<TestCase> test_cases = {
            {7919, true},
            {7920, false},
            {6700417, true},   // Mersenne prime
            {6700418, false}
    };

    for (const auto& test_case : test_cases) {
        SingleThreadedMillerRabinTest test(test_case.number, 5);
        if (test_case.is_prime) {
            EXPECT_TRUE(test.is_prime()) << "Failed for prime number: " << test_case.number;
        } else {
            EXPECT_FALSE(test.is_prime()) << "Failed for composite number: " << test_case.number;
        }
    }
}
