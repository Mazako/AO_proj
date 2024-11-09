add_executable(miller_rabin_gpu_test
        miller_rabin_gpu_test.cpp
        ../miller-cuda/MillerRabinExecutor.cuh
        ../miller-cuda/MillerRabinExecutor.cu
        ../helper/Utils.cu
        ../helper/Utils.h
)

set_target_properties(miller_rabin_gpu_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(miller_rabin_gpu_test PUBLIC
        gtest
        gtest_main
        CUDA::cudart
)

target_include_directories(miller_rabin_gpu_test SYSTEM PUBLIC
        ${CMAKE_SOURCE_DIR}/helper
        ${CMAKE_SOURCE_DIR}/miller-cuda
        ${CMAKE_SOURCE_DIR}/third_party/googletest/googletest/include
)
set(GTest_ROOT ${CMAKE_SOURCE_DIR}/third_party/googletest)
add_dependencies(miller_rabin_gpu_test gtest gtest_main)