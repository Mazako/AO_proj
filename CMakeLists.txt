cmake_minimum_required(VERSION 3.10)
project(MillerRabin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Dodanie GoogleTest
add_subdirectory(third_party/googletest)
message(STATUS "Google Test has been added to the build.")

# Sprawdzenie, czy CUDA jest dostępna
find_package(CUDAToolkit QUIET)
if (CUDAToolkit_FOUND)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 20)
        add_executable(MillerRabin
                MillerRabin.cu                       # Jeśli nadal potrzebujesz CUDA, inaczej usuń to
                helper/Utils.cpp                     # Plik bez CUDA
                helper/Utils.cu                      # Plik dla funkcji CUDA, jeśli potrzebny
                miller-rabin/SingleThreadedMillerRabinTest.cpp
                miller-cuda/MillerRabinExecutor.cu
                miller-cuda/CudaRngWarmup.cu
                )

        set_target_properties(MillerRabin PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                )

        target_link_libraries(MillerRabin CUDA::cudart)
else()
        # Jeśli CUDA nie jest dostępna, budujemy tylko z plikami CPU
        add_executable(MillerRabin
                helper/Utils.cpp
                miller-rabin/SingleThreadedMillerRabinTest.cpp
                helper/Utils.cpp)
endif()

# Ścieżki nagłówków
target_include_directories(MillerRabin PRIVATE
        ${CMAKE_SOURCE_DIR}/helper
        ${CMAKE_SOURCE_DIR}/miller-rabin
        ${CMAKE_SOURCE_DIR}/third_party/googletest/googletest/include
        )

# Dodanie testów
add_subdirectory(tests)
add_dependencies(miller_rabin_test gtest gtest_main)
