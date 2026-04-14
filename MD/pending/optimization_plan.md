# Chess Engine Optimization Plan
## Phase 2: INT8 Quantization · PGO · 2-ply Continuation History

---

## Overview

| # | Optimization | Files | Retraining | Estimated NPS gain |
|---|---|---|---|---|
| A | INT8 L2/L3 quantization | `NNUE.h`, `NNUE.cpp` | No | +20–30% `forwardQ` → ~+8–12% NPS |
| B | Profile-Guided Optimization (PGO) | `ChessEngine.vcxproj` + build process | No | +5–15% NPS (free) |
| C | 2-ply continuation history | `Engine.h`, `Engine.cpp` | No | +2–5% effective depth |

**Recommended order:** A → C → B (PGO last so it profiles the final optimized code)

---

## A · INT8 L2/L3 Quantization

### Why it helps

The current `forwardQ` path:
1. INT16 accumulator → sign-extend to INT32 → convert to float → SCReLU → `float` array (1024 values)
2. L2: `_mm256_fmadd_ps` over 1024 floats → 128 outputs (**16,384 AVX2 FP32 ops**)
3. L3: another `_mm256_fmadd_ps` pass → 64 outputs

AVX2 FP32 processes **8 values/register**. INT8 SIMD (`_mm256_maddubs_epi16`) processes **32 values/register** — 4× the throughput. For the L2 dot product alone this reduces 16,384 ops → ~4,096 ops.

### Architecture of the change

```
Current:  INT16 acc → float32 SCReLU → float32 L2 weights → float32 L2 out
Proposed: INT16 acc → UINT8 clipped → INT8 L2 weights  → INT32 L2 out → float32
```

The SCReLU output is always in [0, 1] after squaring — mapping it to [0, 127] UINT8 loses < 0.4% precision.

### Implementation steps

#### NNUE.h

1. Add quantized L2/L3 weight arrays to `Network::Weights`:
   ```cpp
   // INT8 quantized L2/L3 for fast integer inference
   alignas(64) std::array<std::array<int8_t,  L1_SIZE * 2>, L2_SIZE> L2_weights_i8{};
   alignas(64) std::array<int32_t, L2_SIZE>  L2_biases_i32{};
   alignas(64) std::array<std::array<int8_t,  L2_SIZE>,     L3_SIZE> L3_weights_i8{};
   alignas(64) std::array<int32_t, L3_SIZE>  L3_biases_i32{};
   ```
2. Add two scale constants:
   ```cpp
   static constexpr float L2_QUANT_SCALE = 127.0f;   // SCReLU output → UINT8
   static constexpr float L2_W_SCALE     = 64.0f;    // L2 weight → INT8
   ```

#### NNUE.cpp — `loadWeights()` / `quantize()`

After the existing INT16 L1 quantization block, add:
```cpp
// Quantize L2 weights → INT8
for (int j = 0; j < L2_SIZE; j++) {
    for (int i = 0; i < L1_SIZE * 2; i++) {
        float w = weights.L2_weights[j][i] * L2_W_SCALE;
        weights.L2_weights_i8[j][i] = (int8_t)std::clamp((int)std::round(w), -127, 127);
    }
    weights.L2_biases_i32[j] = (int32_t)std::round(
        weights.L2_biases[j] * L2_QUANT_SCALE * L2_W_SCALE);
}
// Quantize L3 weights → INT8 (same pattern, scale = L3_W_SCALE)
```

#### NNUE.cpp — `forwardQ()`

Replace the float SCReLU + float L2 dot product block:

**Step 1 — Pack INT16 accumulator to UINT8 (replaces float conversion):**
```cpp
alignas(32) uint8_t l1_u8[L1_SIZE * 2];
// White perspective
{
    const int16_t* src = acc.white.data();  // [L1_SIZE]
    for (int i = 0; i < L1_SIZE; i += 16) {
        __m256i v = _mm256_load_si256((__m256i*)(src + i));
        // clamp [0, INT16_MAX], square (SCReLU), scale to [0,127]
        __m256i clamped = _mm256_max_epi16(v, _mm256_setzero_si256());
        clamped = _mm256_min_epi16(clamped, _mm256_set1_epi16(127));
        // pack to uint8 (two 128-bit halves)
        __m128i lo = _mm256_castsi256_si128(clamped);
        __m128i hi = _mm256_extracti128_si256(clamped, 1);
        __m128i packed = _mm_packus_epi16(lo, hi);
        _mm_store_si128((__m128i*)(l1_u8 + i), packed);
    }
}
// Black perspective: l1_u8 + L1_SIZE (same pattern)
```

**Step 2 — INT8 dot product for L2 (replaces float fmadd loop):**
```cpp
alignas(32) float l2Out[L2_SIZE]{};
for (int j = 0; j < L2_SIZE; j++) {
    const int8_t* w = weights.L2_weights_i8[j].data();
    __m256i acc32 = _mm256_setzero_si256();
    for (int i = 0; i < L1_SIZE * 2; i += 32) {
        __m256i inp = _mm256_load_si256((__m256i*)(l1_u8 + i));
        __m256i wgt = _mm256_load_si256((__m256i*)(w + i));
        // _mm256_maddubs_epi16: u8 * s8 → i16 pairs, then hadd
        __m256i prod = _mm256_maddubs_epi16(inp, wgt);
        __m256i prod32 = _mm256_madd_epi16(prod, _mm256_set1_epi16(1));
        acc32 = _mm256_add_epi32(acc32, prod32);
    }
    int32_t sum = hsum_epi32_avx(acc32) + weights.L2_biases_i32[j];
    float val = sum / (L2_QUANT_SCALE * L2_W_SCALE);
    float clamped = std::max(0.0f, std::min(val, 1.0f));
    l2Out[j] = clamped * clamped;  // SCReLU
}
```

Add helper `hsum_epi32_avx` (horizontal sum of 8×INT32 AVX register) alongside the existing `hsum_avx`.

Apply the same pattern to L3 (L2→L3 is 128×64, smaller but same approach).

### Validation

Run the SmokeTests suite — the quantized path should produce eval scores within ±2 centipawns of the float path on all test positions. If drift is larger, increase `L2_W_SCALE` (up to 127).

---

## B · Profile-Guided Optimization (PGO)

### Why it helps

MSVC's PGO instruments every function call, branch, and indirect jump. On the optimized rebuild it uses real execution data to:
- Inline hot functions more aggressively (esp. the move generation inner loop)
- Lay out hot code paths contiguously in memory (fewer instruction cache misses)
- Optimize branch prediction for frequent cases (e.g., cut nodes, hash hits)
- Improve register allocation in the alpha-beta loop

No code changes required — this is purely a build-process change.

### Steps

#### Step 1 — Instrument build
In Visual Studio:
1. Select **Release x64** configuration
2. `Build` → `Profile Guided Optimization` → `Instrument`  
   *(or set `ChessEngine.vcxproj` property: `Link > Profile Guided Database = ChessEngine.pgd`  and `Linker > Optimization > Profile Guided Optimization = PGI`)*
3. Build — produces `ChessEngine.exe` with instrumentation hooks + `ChessEngine.pgd`

#### Step 2 — Collect profile data
Run the instrumented binary with a representative workload. **Best option:** run a perft or a fixed-time game against Stockfish for ~60 seconds:
```
ChessEngine.exe bench 15 1 20 default time
```
Or use the existing `PerftTests.exe` suite — it exercises the full search + eval path.

The `.pgc` profile data file is written automatically to the executable's directory.

#### Step 3 — Optimized rebuild
Back in Visual Studio:
1. `Build` → `Profile Guided Optimization` → `Optimize`  
   *(or set linker flag: `LTCG:PGO`)*
2. Rebuild — the linker merges `.pgc` data into the final binary

#### Step 4 — Verify
Compare NPS before/after on the same perft position:
```
go perft 6    (starting position, expected 119,060,324 nodes)
```

### Note on future changes
PGO profiles become stale when code changes significantly. Re-run the PGO process after implementing optimization A and C.

---

## C · 2-ply Continuation History

### Why it helps

The engine currently scores quiet moves using:
- `history_[color][from][to]` — general quiet history
- `pieceToHistory_[color][piece][to]` — piece-destination pattern
- `counterMoveHist_[prevPiece][prevTo][to]` — 1-ply: what happened after the opponent's last move

**2-ply continuation history** adds: what happened two plies ago — effectively asking *"when we moved this piece here, and then the opponent played that, what did we play next?"* This is the same technique Stockfish calls `contHist[1]` (ply-2 follow-up table).

In practice it provides a small but consistent improvement to move ordering at depths 5+, leading to more beta cutoffs and ~3% more effective depth in typical middlegames.

### Implementation

#### Engine.h

Add the table and a second `previousMoves_` lookup (ply-2 is already accessible via `previousMoves_[ply - 2]`):
```cpp
// 2-ply continuation history [prevPrevPieceIdx][prevPrevToSq][curToSq]
int contHist2_[12][64][64]{};
```
Add to `clearSearchState()` and `clearBetweenMoves()`:
```cpp
std::memset(contHist2_, 0, sizeof(contHist2_));
```
Age between iterations (same as `counterMoveHist_`):
```cpp
for (auto& a : contHist2_) for (auto& b : a) for (auto& v : b) v /= 2;
```

#### Engine.cpp — `scoreMove()`

After the existing `counterMoveHist_` bonus (around line 1006), add:
```cpp
// 2-ply continuation history bonus
if (ply >= 2 && previousMoves_[ply - 2].from.isValid()) {
    Piece pp = board.getPiece(previousMoves_[ply - 2].to);
    if (!pp.isNone()) {
        int ppIdx = ((int)pp.type - 1) * 2 + (int)pp.color;
        int ppTo  = previousMoves_[ply - 2].to.rank * 8 + previousMoves_[ply - 2].to.col;
        score += contHist2_[ppIdx][ppTo][toSq] / 2;  // half weight vs 1-ply
    }
}
```

#### Engine.cpp — History update block (around line 1739)

After the existing `counterMoveHist_` update, add:
```cpp
// Update 2-ply continuation history
if (ply >= 2 && previousMoves_[ply - 2].from.isValid()) {
    Piece pp2 = board.getPiece(previousMoves_[ply - 2].to);
    if (!pp2.isNone()) {
        int ppIdx2 = ((int)pp2.type - 1) * 2 + (int)pp2.color;
        int ppTo2  = previousMoves_[ply - 2].to.rank * 8 + previousMoves_[ply - 2].to.col;
        int& entry = contHist2_[ppIdx2][ppTo2][curTo];
        int bonus  = std::min(depth * depth, 400);
        entry = std::clamp(entry + bonus, -8000, 8000);
        // malus for all other quiets tried before the cutoff move
        for (int k = 0; k < quietsTried; k++) {
            int mTo = quietsTriedList[k].to.rank * 8 + quietsTriedList[k].to.col;
            int& malEntry = contHist2_[ppIdx2][ppTo2][mTo];
            malEntry = std::clamp(malEntry - bonus / 2, -8000, 8000);
        }
    }
}
```

### Tuning note

The half-weight divisor (`/ 2` in scoreMove) is a starting point. After implementation, test with `/ 3` and `/ 4` using self-play games (500+ games, 5+0.05 time control) to find the best value for your net.

---

## Combined Expected Impact

| After implementing | Expected NPS vs baseline |
|---|---|
| A only | +8–12% |
| A + C | +10–15% |
| A + C + B | +16–28% |
| On top of #1+#2+#3 (already done) | Cumulative |

The combined effect of all 6 optimizations (3 already done + these 3) should bring the NNUE bot from depth 8–9 to **depth 11–12**, closing most of the gap with the classic bot.
