cmake_minimum_required(VERSION 3.29)
set(CUDA_ARCHITECTURES 70)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.6/bin/nvcc")
project(MillerRabin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

# Dodanie GoogleTest
add_subdirectory(third_party/googletest)
message(STATUS "Google Test has been added to the build.")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

include_directories(
        helper
        miller-rabin
        miller-cuda/miller-rabin
        miller-cuda
)

add_executable(MillerRabin MillerRabin.cu
        helper/Utils.cu
        helper/Utils.h
        miller-rabin/SingleThreadedMillerRabinTest.cpp
        miller-rabin/SingleThreadedMillerRabinTest.h
        miller-cuda/MillerRabinExecutor.cu
        miller-cuda/MillerRabinExecutor.cuh
        miller-cuda/CudaRngWarmup.cuh
        miller-cuda/CudaRngWarmup.cu
        miller-rabin/MultiThreadedMillerRabinTest.cpp
        miller-rabin/MultiThreadedMillerRabinTest.h
        helper/FileLoader.cpp
        helper/FileLoader.h
        helper/TestRunner.cpp
        helper/TestRunner.h
        miller-cuda/MillerRabinMultipleNumberExecutor.cuh
        miller-cuda/MillerRabinMultipleNumberExecutor.cu
)

set_target_properties(MillerRabin PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

find_package(CUDAToolkit)
target_link_libraries(MillerRabin CUDA::cudart)

# Ścieżki nagłówków
target_include_directories(MillerRabin PRIVATE
        ${CMAKE_SOURCE_DIR}/helper
        ${CMAKE_SOURCE_DIR}/miller-rabin
        ${CMAKE_SOURCE_DIR}/third_party/googletest/googletest/include
)

# Dodanie testów
add_subdirectory(tests)
add_subdirectory(tests-cuda)
add_dependencies(miller_rabin_test miller_rabin_gpu_test gtest gtest_main)

target_compile_options(MillerRabin PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        >)
