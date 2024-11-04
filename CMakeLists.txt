cmake_minimum_required(VERSION 3.10)
project(MillerRabin)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(third_party/googletest)
message(STATUS "Google Test has been added to the build.")


add_executable(MillerRabin
        MillerRabin.cpp
        helper/Utils.cpp
        miller-rabin/SingleThreadedMillerRabinTest.cpp
        )


target_include_directories(MillerRabin PRIVATE
        ${CMAKE_SOURCE_DIR}/helper
        ${CMAKE_SOURCE_DIR}/miller-rabin
        ${CMAKE_SOURCE_DIR}/third_party/googletest/googletest/include
        )

add_subdirectory(tests)

add_dependencies(miller_rabin_test gtest gtest_main)