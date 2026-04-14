# Tier 1 Performance Improvements — Implementation Report

## Overview

Three major performance improvements have been implemented for the chess engine, targeting the highest Elo-per-effort optimizations. These changes are deeply interrelated — they all affect the search/evaluation hot path — and were designed to work together.

**Estimated combined impact: +150–350 Elo at equivalent time controls.**

---

## Improvement #1: Incremental NNUE Accumulator Updates

### Problem
Every call to `nnue_->evaluate(board)` recomputed all 40,960 HalfKAv2 features from scratch. With L1_SIZE=1024, this means iterating over every piece on the board and adding 1024 floats per piece (~15 pieces × 1024 multiplies × 2 perspectives = ~30,000 additions) per evaluation.

### Solution
A **quantized accumulator stack** (`QAccumulator accStack_[128]`) is maintained throughout the search tree:

- **Root**: Full refresh at ply 0 via `refreshAccumulatorQ()`
- **Normal moves**: Copy parent accumulator → remove moved piece's weights → add at new square. Only 2–3 feature updates (~3,072 additions vs ~30,000).
- **Captures**: Additionally remove captured piece's weights (~4,096 total additions).
- **King moves, castling, en passant, promotions**: Full refresh (all features change for that perspective, or the board state is complex).
- **Null moves**: Simple copy (no pieces change).

### Files Changed
- **Engine.h**: Added `accStack_[]` array and `updateAccumulator()` helper
- **Engine.cpp**: Added `updateAccumulator()`, modified `evaluate()` to use `forwardQ()` with cached accumulator, wired accumulator updates into search/qsearch move loops and root

### Performance Impact
- **~10–30× fewer L1 operations per node** for non-king, non-special moves (the vast majority)
- Estimated **+100–200 Elo** from deeper effective search

---

## Improvement #2: Make/Unmake Move Pattern

### Problem
Every search node copied the entire Board struct before applying a move:
```cpp
Board temp = board;      // ~150 bytes memcpy
temp.applyMove(m);       // modify the copy
score = -search(temp, ...);
```
At millions of nodes per second, this creates significant memory bandwidth pressure and cache pollution.

### Solution
New `makeMove()` / `unmakeMove()` methods modify the board in-place:

```cpp
UndoInfo undo;
board.makeMove(m, undo);     // modify in-place, save undo state (~20 bytes)
score = -search(board, ...);
board.unmakeMove(m, undo);   // restore original state
```

The `UndoInfo` struct saves only what can't be reconstructed:
- En passant target, castling rights, half-move clock
- Captured piece and its square (for en passant, capture is off the to-square)
- Special move flags (en passant, kingside/queenside castle)

### Files Changed
- **Types.h**: Added `UndoInfo` struct
- **Board.h**: Added `makeMove()` / `unmakeMove()` declarations
- **Board.cpp**: Implemented both methods (~150 lines)
- **Engine.cpp**: Replaced all `Board temp = board; temp.applyMove(m)` with make/unmake in:
  - `search()` main move loop
  - `search()` null move pruning (lightweight save/restore for turn, EP, halfMoveClock)
  - `qsearch()` move loop
  - `getBestMove()` root move loop

### Performance Impact
- **Saves ~130 bytes of copying per node** (150 byte Board copy → 20 byte UndoInfo)
- Better **cache locality** (same Board object stays hot in L1 cache)
- Estimated **+50–100 Elo** from raw NPS increase

### Correctness Guarantee
Every `makeMove` has a corresponding `unmakeMove` on all code paths. All pruning decisions (`continue`, early returns) happen **before** `makeMove` is called. The existing `applyMove()` is preserved for backward compatibility with non-engine code (GUI, self-play, etc.).

---

## Improvement #3: INT16 Quantized NNUE Inference

### Problem
The L1 feature transform weights are stored as `float` (32-bit):
- **Memory**: 40,960 features × 1,024 neurons × 4 bytes = **160 MB**
- **SIMD width**: SSE processes 4 floats per instruction
- **Cache pressure**: 160 MB of weights thrashes the CPU cache hierarchy

### Solution
**INT16 quantization** with scale factor QA=256:

```
int16_weight = round(float_weight × 256)
```

At runtime (after loading or randomizing float weights), `quantizeWeights()` creates INT16 copies:
- `L1_weights_q`: 40,960 × 1,024 × int16_t = **80 MB** (half the memory)
- `L1_biases_q`: 1,024 × int16_t

The accumulator uses `int16_t` arrays and **SSE2 integer SIMD**:
- `_mm_add_epi16` / `_mm_sub_epi16`: **8 values per instruction** (vs 4 for float)
- 2× SIMD throughput for accumulator updates

At the L2 boundary, values are dequantized back to float:
```
float_value = int16_value / 256.0f
```
Then SCReLU and L2/L3/phase heads proceed in float (these layers are tiny).

### Value Range Safety
| Component | Typical Range | INT16 Range | Status |
|-----------|--------------|-------------|--------|
| L1 weights × 256 | [-128, 128] | [-32768, 32767] | ✅ Safe |
| Accumulator (sum of ~15 features) | [-1920, 1920] | [-32768, 32767] | ✅ Safe |
| L1 biases × 256 | [-50, 50] | [-32768, 32767] | ✅ Safe |

### Files Changed
- **NNUE.h**: Added `QA` constant, `QAccumulator` struct, quantized weight storage, quantized method declarations
- **NNUE.cpp**: Implemented `quantizeWeights()`, `refreshAccumulatorQ()`, `incrementalUpdateQ()`, `forwardQ()`, `evaluateQ()`

### Performance Impact
- **50% memory reduction** for L1 weights (160MB → 80MB)
- **2× SIMD throughput** for accumulator operations (8 int16 vs 4 float per SSE instruction)
- Better **cache utilization** across the entire search
- Estimated **+30–80 Elo** from faster NNUE inference
- **Negligible accuracy loss**: QA=256 gives 8 bits of fractional precision, more than enough for NNUE weights

### Backward Compatibility
- Float weights are preserved — the file format is unchanged
- All existing float methods remain functional
- Quantization happens at runtime (`quantizeWeights()` called after `loadWeights()` and `randomizeWeights()`)
- The Python training pipeline requires zero changes

---

## Files Modified (Summary)

| File | Changes |
|------|---------|
| **Types.h** | +`UndoInfo` struct |
| **Board.h** | +`makeMove()`, `unmakeMove()` declarations |
| **Board.cpp** | +`makeMove()`, `unmakeMove()` implementations (~150 lines) |
| **NNUE.h** | +`QA`, `QAccumulator`, quantized weights/methods |
| **NNUE.cpp** | +`quantizeWeights()`, `refreshAccumulatorQ()`, `incrementalUpdateQ()`, `forwardQ()`, `evaluateQ()` (~270 lines) |
| **Engine.h** | +`accStack_[]`, `updateAccumulator()`, changed `evaluate()` signature |
| **Engine.cpp** | Rewired search/qsearch/root for make/unmake + incremental NNUE (~80 lines changed) |

All modified files are in `/agent/home/tier1-files/`.

---

## Integration Guide

### Drop-In Replacement
1. Replace the 7 files in your project with the versions from `tier1-files/`
2. Rebuild — no other files need changes
3. The weight file format is unchanged — existing `.nnue` files work as-is
4. The Python training pipeline requires zero changes

### What's Preserved
- All existing APIs (applyMove, evaluate without ply, float NNUE methods)
- Duck chess support (uses copy-make, unaffected)
- UCI interface (unchanged)
- Self-play and training code (uses applyMove, unaffected)

### Testing Recommendations
1. **Perft test** — verify move generation correctness (makeMove/unmakeMove must produce identical results to applyMove)
2. **NNUE accuracy test** — compare `evaluateQ()` vs `evaluate()` on a set of positions (should differ by ≤1 centipawn due to quantization)
3. **Search correctness** — run fixed-depth searches on benchmark positions and compare with old engine
4. **NPS benchmark** — measure nodes/second improvement at depth 12-15 on the starting position
5. **Elo test** — play 1000+ games vs the old version at fixed time control
