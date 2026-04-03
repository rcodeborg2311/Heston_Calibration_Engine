cmake_minimum_required(VERSION 3.20)
project(heston_engine CXX)

# ---------------------------------------------------------------------------
# Language standard
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Build-type flags
# ---------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Common warnings — must compile clean
add_compile_options(-Wall -Wextra -Wpedantic)

# Release: full optimisation
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -DNDEBUG -flto")

# Debug: sanitizers
set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g -fsanitize=address,undefined")

# ---------------------------------------------------------------------------
# Find FFTW3 (system install: apt install libfftw3-dev | brew install fftw)
# ---------------------------------------------------------------------------
find_library(FFTW3_LIBRARY NAMES fftw3 REQUIRED)
find_path(FFTW3_INCLUDE_DIR NAMES fftw3.h REQUIRED)
if(NOT FFTW3_LIBRARY OR NOT FFTW3_INCLUDE_DIR)
    message(FATAL_ERROR
        "FFTW3 not found.\n"
        "  Linux: sudo apt install libfftw3-dev\n"
        "  macOS: brew install fftw\n")
endif()
message(STATUS "Found FFTW3: ${FFTW3_LIBRARY}")
message(STATUS "FFTW3 include: ${FFTW3_INCLUDE_DIR}")

# ---------------------------------------------------------------------------
# FetchContent: Catch2 v3 (tests), nanobench (benchmarks)
# ---------------------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.3
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(Catch2)

FetchContent_Declare(
    nanobench
    GIT_REPOSITORY https://github.com/martinus/nanobench.git
    GIT_TAG        v4.3.11
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(nanobench)

# ---------------------------------------------------------------------------
# heston_lib — shared library of all pricing components
# ---------------------------------------------------------------------------
add_library(heston_lib STATIC
    src/characteristic_fn.cpp
    src/carr_madan.cpp
    src/adi_solver.cpp
    src/calibrator.cpp
    src/greeks.cpp
    src/vol_surface.cpp
)

target_include_directories(heston_lib
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${FFTW3_INCLUDE_DIR}
)

target_link_libraries(heston_lib
    PUBLIC
        ${FFTW3_LIBRARY}
)

# Treat all warnings as errors in library code
target_compile_options(heston_lib PRIVATE -Werror)

# ---------------------------------------------------------------------------
# heston_demo — main CLI executable
# ---------------------------------------------------------------------------
add_executable(heston_demo src/main.cpp)
target_link_libraries(heston_demo PRIVATE heston_lib)

# ---------------------------------------------------------------------------
# bench_pricing — nanobench benchmarks
# ---------------------------------------------------------------------------
add_executable(bench_pricing bench/bench_pricing.cpp)
target_include_directories(bench_pricing PRIVATE ${nanobench_SOURCE_DIR}/src/include)
target_link_libraries(bench_pricing PRIVATE heston_lib)

# ---------------------------------------------------------------------------
# heston_tests — Catch2 test suite
# ---------------------------------------------------------------------------
add_executable(heston_tests
    tests/test_characteristic_fn.cpp
    tests/test_carr_madan.cpp
    tests/test_adi_solver.cpp
    tests/test_calibrator.cpp
    tests/test_greeks.cpp
)
target_link_libraries(heston_tests
    PRIVATE
        heston_lib
        Catch2::Catch2WithMain
)

# Register with CTest
include(CTest)
include(Catch)
catch_discover_tests(heston_tests)

# ---------------------------------------------------------------------------
# Install (optional)
# ---------------------------------------------------------------------------
install(TARGETS heston_demo heston_tests DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
