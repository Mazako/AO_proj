cmake_minimum_required(VERSION 3.29)
project(MillerRabin)

set(CMAKE_CXX_STANDARD 20)

add_executable(MillerRabin MillerRabin.cpp
        helper/Utils.cpp
        helper/Utils.h
        miller-rabin/SingleThreadedMillerRabinTest.cpp
        miller-rabin/SingleThreadedMillerRabinTest.h
)
