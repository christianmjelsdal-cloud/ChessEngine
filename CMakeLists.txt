cmake_minimum_required(VERSION 3.14)
project(ChessEngine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# AUDIT FIX BT-2: Default to Release build with optimizations
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
if(MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)  # FIX 12.8: suppress C4996 warnings for portable code
    add_compile_definitions(NOMINMAX)                  # FIX 12.9: prevent Windows min/max macros
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /DNDEBUG")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
endif()

# AUDIT FIX BT-1: Portable SIMD flags
# NOTE: AVX2 is enabled by default. If targeting pre-2013 CPUs, set -DUSE_AVX2=OFF.
# Runtime CPUID check is recommended in main() for portable builds.
option(USE_AVX2 "Enable AVX2 SIMD" ON)
if(USE_AVX2)
    if(MSVC)
        add_compile_options(/arch:AVX2)
    else()
        add_compile_options(-mavx2 -mfma -msse4.1)
    endif()
endif()

# ── Threading (portable) ─────────────────────────────────────────────
find_package(Threads REQUIRED)


# ── Syzygy tablebase support (optional, requires Fathom) ─────────────
# Set -DUSE_SYZYGY=ON at cmake configure time to enable probing.
# Fathom files (tbprobe.h / tbprobe.c) must be present in the source tree.
option(USE_SYZYGY "Enable Syzygy tablebase probing via Fathom" ON)
# ARCHITECTURE FIX M-6: Training code separated from engine binary
# ── Core chess library (no GUI dependencies) ──────────────────────────
add_library(chess_core STATIC
    Board.cpp
    MoveGen.cpp
    Bitboard.cpp
    Engine.cpp
    NNUE.cpp
    UCI.cpp
    GameLogic.cpp
    Syzygy.cpp
)
# ── Training library (self-play & NNUE training, not needed by UCI engine) ──
add_library(chess_training STATIC
    SelfPlayGen.cpp
    NNUETrainer.cpp
)
target_link_libraries(chess_training PUBLIC chess_core)
if(USE_SYZYGY AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tbprobe.c")
    target_sources(chess_core PRIVATE tbprobe.c)
    target_compile_definitions(chess_core PUBLIC HAS_SYZYGY)
    message(STATUS "Syzygy tablebase support ENABLED")
else()
    message(STATUS "Syzygy tablebase support DISABLED (pass -DUSE_SYZYGY=ON and ensure tbprobe.c is present)")
endif()
target_include_directories(chess_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(chess_core PRIVATE Threads::Threads)

# AUDIT FIX BT-3: Standalone UCI engine executable
add_executable(chess_engine main.cpp)
target_link_libraries(chess_engine PRIVATE chess_core)
target_link_libraries(chess_engine PRIVATE Threads::Threads)

# ── Test utilities (shared by both test executables) ──────────────────
# Compile Test.cpp with TEST_NO_MAIN to exclude its main() — provides
# perft() and validateBoardConsistency() to any test that links this.
add_library(test_utils STATIC Test.cpp)
target_compile_definitions(test_utils PRIVATE TEST_NO_MAIN)
target_link_libraries(test_utils PUBLIC chess_core)

# ── Perft test executable ─────────────────────────────────────────────
add_executable(test_exe Test.cpp)
target_link_libraries(test_exe PRIVATE chess_core Threads::Threads)

# ── Smoke test executable ─────────────────────────────────────────────
add_executable(smoketest_exe SmokeTest.cpp)
target_link_libraries(smoketest_exe PRIVATE test_utils Threads::Threads)

# TODO: Add test targets for NNUE roundtrip, training format compatibility,
# and evaluation consistency (incremental vs full refresh).
