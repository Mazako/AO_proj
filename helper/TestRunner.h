#ifndef TESTRUNNER_H
#define TESTRUNNER_H

#include <vector>
#include <cstdint>
#include <string>

class TestRunner {
public:
    static void run_tests(const std::vector<uint64_t>& numbers, int iterations, const std::string& mode);
};

#endif // TESTRUNNER_H
