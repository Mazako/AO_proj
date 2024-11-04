add_executable(miller_rabin_test
        miller_rabin_test.cpp
        ../miller-rabin/SingleThreadedMillerRabinTest.cpp
        ../helper/Utils.cpp
        )


target_link_libraries(miller_rabin_test PUBLIC gtest gtest_main)

target_include_directories(miller_rabin_test SYSTEM PUBLIC
        ${CMAKE_SOURCE_DIR}/third_party/googletest/googletest/include
        ${CMAKE_SOURCE_DIR}/helper
        ${CMAKE_SOURCE_DIR}/miller-rabin
        )
set(GTest_ROOT ${CMAKE_SOURCE_DIR}/third_party/googletest)
