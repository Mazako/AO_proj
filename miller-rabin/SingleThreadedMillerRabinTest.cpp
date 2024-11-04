#include "SingleThreadedMillerRabinTest.h"
#include "Utils.h"
#include "gtest/gtest.h"

SingleThreadedMillerRabinTest::SingleThreadedMillerRabinTest(uint64_t number_to_test, int iterations)
    : number(number_to_test), iterations(iterations) {}

bool SingleThreadedMillerRabinTest::is_prime() const {
    if (number <= 1 || (number > 2 && number % 2 == 0))
        return false;
    if (number == 2)
        return true;

    return miller_rabin_primality_test();
}

bool SingleThreadedMillerRabinTest::miller_rabin_primality_test() const {
    uint64_t exponent_of_two, odd_part;
    Utils::decompose_number(number - 1, exponent_of_two, odd_part);

    for (int i = 0; i < iterations; ++i) {
        uint64_t base = Utils::get_random_number(2, number - 2);
        uint64_t x = Utils::modular_pow(base, odd_part, number);

        if (x == 1 || x == number - 1)
            continue;

        bool composite = true;
        for (uint64_t j = 0; j < exponent_of_two - 1; ++j) {
            x = Utils::modular_pow(x, 2, number);
            if (x == number - 1) {
                composite = false;
                break;
            }
        }

        if (composite)
            return false;
    }
    return true;
}
