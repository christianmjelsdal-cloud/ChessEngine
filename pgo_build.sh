#!/bin/bash
# PGO (Profile-Guided Optimization) build script for GCC/Clang
# Typically gives 10-15% speedup for search-heavy code.
#
# Usage:
#   ./pgo_build.sh [gcc|clang]
#
# Requirements:
#   - GCC 10+ or Clang 12+
#   - CMake (uses CMakeLists.txt)

set -e

COMPILER=${1:-gcc}

# FIX 12.15: Validate compiler input
if [ "$COMPILER" != "gcc" ] && [ "$COMPILER" != "clang" ]; then
    echo "Error: Invalid compiler '$COMPILER'. Must be 'gcc' or 'clang'."
    exit 1
fi

BUILD_DIR="build-pgo"
BINARY="ChessEngine"

echo "=== PGO Build ($COMPILER): Phase 1 — Instrumented Build ==="

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

if [ "$COMPILER" = "clang" ]; then
    CC=clang CXX=clang++ cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS="-fprofile-instr-generate -O2 -DNDEBUG -march=native" \
        -DCMAKE_EXE_LINKER_FLAGS="-fprofile-instr-generate"
else
    CC=gcc CXX=g++ cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS="-fprofile-generate -O2 -DNDEBUG -march=native" \
        -DCMAKE_EXE_LINKER_FLAGS="-fprofile-generate"
fi

make -j$(nproc)

echo ""
echo "=== PGO Build ($COMPILER): Phase 2 — Training Run ==="
echo "Running self-play to collect profile data..."

./$BINARY --generate --games 100 --depth 5 --workers 1

echo ""
echo "=== PGO Build ($COMPILER): Phase 3 — Optimized Build ==="

if [ "$COMPILER" = "clang" ]; then
    # Merge profile data
    llvm-profdata merge -output=default.profdata default.profraw
    
    # FIX 12.2: Save profdata outside build dir before deleting it
    cp default.profdata /tmp/chess_pgo.profdata
    cd ..
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    CC=clang CXX=clang++ cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS="-fprofile-instr-use=/tmp/chess_pgo.profdata -O2 -DNDEBUG -march=native" \
        -DCMAKE_EXE_LINKER_FLAGS="-fprofile-instr-use=/tmp/chess_pgo.profdata"
else
    cd ..
    # NOTE: BUILD_DIR is reassigned here. The final echo path is relative to the new BUILD_DIR.
    rm -rf "${BUILD_DIR}-opt"
    mkdir -p "${BUILD_DIR}-opt"
    cd "${BUILD_DIR}-opt"
    
    CC=gcc CXX=g++ cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS="-fprofile-use=../$BUILD_DIR -O2 -DNDEBUG -march=native -fprofile-correction" \
        -DCMAKE_EXE_LINKER_FLAGS="-fprofile-use=../$BUILD_DIR"
    
fi

make -j$(nproc)

echo ""
echo "=== PGO Build Complete ==="
echo "Optimized binary: $(pwd)/$BINARY"  # FIX 12.3: use actual CWD instead of potentially-shadowed variable
echo "Expected speedup: 10-15% for search operations"
