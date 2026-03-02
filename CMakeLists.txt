cmake_minimum_required(VERSION 3.21)
project(agi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(oracle src/main.cpp)
target_compile_features(oracle PRIVATE cxx_std_17)
target_include_directories(oracle PRIVATE ${CMAKE_SOURCE_DIR}/src)

enable_testing()
add_subdirectory(tests)
