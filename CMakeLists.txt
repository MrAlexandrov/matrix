cmake_minimum_required(VERSION 3.14)
project(matrix CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
endif()

add_library(MatrixLib INTERFACE)

target_include_directories(MatrixLib INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

enable_testing()
add_subdirectory(tests)

set(CTEST_BINARY_DIRECTORY ${CMAKE_BINARY_DIR})
