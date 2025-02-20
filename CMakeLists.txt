cmake_minimum_required(VERSION 3.14)
project(matrix CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
# endif()

find_package(OpenMP REQUIRED)

# if(OpenMP_CXX_FOUND)
#     message(STATUS "Found OpenMP: ${OpenMP_CXX_LIBRARIES}")
#     set(OpenMP_CXX_FLAGS ${OpenMP_CXX_FLAGS} ${OpenMP_CXX_FLAGS})
#     set(OpenMP_CXX_LIBRARIES ${OpenMP_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
# endif()

# if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#     message(STATUS "Detected Clang, adjusting OpenMP flags...")
#     set(OpenMP_CXX_FLAGS -Xpreprocessor -fopenmp)
#     set(OpenMP_CXX_LIBRARIES -lomp)
# endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mavx2 -mfma -O3 -pthread")
endif()

add_library(MatrixLib INTERFACE)

target_include_directories(MatrixLib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# target_compile_options(MatrixLib INTERFACE ${OpenMP_CXX_FLAGS})
target_compile_options(MatrixLib INTERFACE -march=native -mavx2 -mfma -msse3)
# target_link_libraries(MatrixLib INTERFACE OpenMP::OpenMP_CXX)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} MatrixLib)

enable_testing()
add_subdirectory(tests)

set(CTEST_BINARY_DIRECTORY ${CMAKE_BINARY_DIR})
