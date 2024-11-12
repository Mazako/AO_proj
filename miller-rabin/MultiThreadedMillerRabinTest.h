#ifndef MULTITHREADEDMILLERRABINTEST_H
#define MULTITHREADEDMILLERRABINTEST_H
#include <cstdint>

class MultiThreadedMillerRabinTest {
public:
    MultiThreadedMillerRabinTest(uint64_t number_to_test, int iterations);
    bool is_prime() const;

private:
    uint64_t number;
    int iterations;
};

#endif //MULTITHREADEDMILLERRABINTEST_H
