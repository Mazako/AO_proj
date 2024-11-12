#include "MultiThreadedMillerRabinTest.h"
#include "SingleThreadedMillerRabinTest.h"
#include <future>
#include <vector>



MultiThreadedMillerRabinTest::MultiThreadedMillerRabinTest(uint64_t number_to_test, int iterations)
    : number(number_to_test), iterations(iterations) {}

bool MultiThreadedMillerRabinTest::is_prime() const {
    if (number <= 1 || (number > 2 && number % 2 == 0))
        return false;
    if (number == 2)
        return true;

    int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 2;

    auto test_prime = [this](int local_iterations) {
        SingleThreadedMillerRabinTest single_test(this->number, local_iterations);
        return single_test.is_prime();
    };

    int iterations_per_thread = iterations / num_threads;
    int remaining_iterations = iterations % num_threads;

    std::vector<std::future<bool>> futures;

    for (int i = 0; i < num_threads; ++i) {
        int iters = iterations_per_thread + (i < remaining_iterations ? 1 : 0);
        futures.emplace_back(std::async(std::launch::async, test_prime, iters));
    }

    for (auto& future : futures) {
        if (!future.get()) {
            return false;
        }
    }
    return true;
}