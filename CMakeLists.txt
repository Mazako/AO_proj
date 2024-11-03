cmake_minimum_required(VERSION 3.29)
project(MillerRabin LANGUAGES CUDA CXX)

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
)

set_target_properties(MillerRabin PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

find_package(CUDAToolkit)
target_link_libraries(MillerRabin CUDA::cudart)
