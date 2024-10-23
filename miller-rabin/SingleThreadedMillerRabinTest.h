#ifndef SINGLE_THREADED_MILLER_RABIN_TEST_H
#define SINGLE_THREADED_MILLER_RABIN_TEST_H

#include <cstdint>

class SingleThreadedMillerRabinTest {
public:
    explicit SingleThreadedMillerRabinTest(uint64_t number_to_test, int iterations = 5);
    bool is_prime() const;

private:
    uint64_t number;
    int iterations;

    bool miller_rabin_primality_test() const;
};

#endif
