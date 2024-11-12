#ifndef TESTRUNNER_H
#define TESTRUNNER_H

#include <cstdint>
#include <string>

class TestRunner {
public:
    static void run_tests(uint64_t* numbers, int count, int iterations, const std::string& mode);
};

#endif // TESTRUNNER_H
