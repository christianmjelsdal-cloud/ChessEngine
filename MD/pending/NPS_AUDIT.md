# Duck Chess NPS Audit

**Last updated:** April 16, 2026  
**Baseline:** ~300 NPS (depth 3, 12 workers) before this session  
**Current:** ~1600+ NPS (depth 2, 12 workers) after fixes this session

---

## What Was Fixed This Session

| Fix | Impact |
|---|---|
| Root accumulator never seeded — all evals fell back to HCE | ~2-3x |
| `forwardQ` SCReLU dequantize was scalar (512 iters) → AVX2 | ~3-4x |
| `refreshAccumulatorQ` was scalar → AVX2 | ~1.3x |
| `getDuckPlacements` + `orderDuckPlacements` called per chess move → hoisted | ~1.3x |
| `maxDucks` caps tightened (16→12 at depth 3, 10→7 at depth 2) | ~1.25x |
| `orderMoves`/`orderDuckPlacements` heap alloc → stack array + insertion sort | ~5-10% |
| `scoreDuckPlacement` 64-sq king scan → `board.whiteKingSq`/`blackKingSq` | ~10-15% |
| `scoreDuckPlacement` 64-sq slider scan → bitboard popcount | ~10-15% |
| NPS reporting: cumulative average → interval EWMA (display fix, not speed) | — |

---

## Remaining Bottlenecks

### 1. `*myAcc = postChess` — 420KB memcpy per node (HIGHEST PRIORITY)

**Location:** `Engine.cpp` — duck placement inner loop  
**Cost:** `QAccumulator` = 512 × 2 × int16 = 2KB. Copied once per duck placement per chess move.  
At depth 2: 7 ducks × 30 chess moves = 210 copies × 2KB = **420KB per node**.  
At depth 3: 12 ducks × 30 chess moves = 360 copies × 2KB = **720KB per node**.

**Root cause:** `searchDuck` at ply+1 reads `accStack[ply]` as its parent. The only way to communicate the post-duck accumulator to the child is to write it into `accStack[ply]` before recursing. Since `postChess` (the post-chess scratch) and `myAcc` (`accStack[ply]`) are different memory locations, a copy is required.

**Fix requires restructuring:**  
Pass the accumulator as a direct pointer parameter to `searchDuck` instead of reading it by index from `accStack`. Signature change:
```cpp
int searchDuck(Board& board, int depth, int alpha, int beta, int ply,
               DuckNNUE::QAccumulator* accStack,
               DuckNNUE::QAccumulator* parentAcc);  // <-- add this
```
Then the duck loop applies the delta to `postChess` in-place and passes `&postChess` directly as `parentAcc` to the recursive call. Eliminates all 210 copies — only the 30 `postChess = *parentAcc` copies remain (one per chess move, unavoidable).

**Estimated gain:** ~30-40% NPS improvement.

---

### 2. `postChess = *parentAcc` — 60KB memcpy per node

**Location:** `Engine.cpp` — chess move loop  
**Cost:** 2KB × 30 chess moves = 60KB per node.

**Root cause:** Each chess move needs its own post-chess accumulator as a base for the duck delta loop. The parent accumulator must be preserved intact for the next chess move, so a copy is required.

**Partial fix (no restructuring):**  
Use AVX2 `memcpy` explicitly (compiler may already do this, but explicit `_mm256_loadu_si256`/`_mm256_storeu_si256` loop guarantees it). The arrays are `alignas(32)` so aligned stores are possible.

**Full fix requires restructuring:**  
Maintain a "dirty" flag and undo the chess delta after each chess move instead of copying. This turns the copy into 2-3 `removeFeatureQ`/`addFeatureQ` calls (AVX2, ~512 ops each) which is cheaper than a 2KB memcpy.

**Estimated gain:** ~10-15% NPS improvement.

---

### 3. `see()` called for every capture in `scoreMove` for duck chess

**Location:** `Engine.cpp` — `scoreMove()`  
**Cost:** Full SEE traversal (iterative LVA scan) for every capture. In duck chess there is no check concept, so SEE is less meaningful — a simpler MVV-LVA score would be faster and equally effective for ordering.

**Fix (no restructuring needed):**  
Add a duck chess fast path in `scoreMove`:
```cpp
if (board.isDuckChess && isCapture(board, m)) {
    // MVV-LVA: victim value * 10 - attacker value (no SEE needed)
    return 5000000 + mvvLva(board, m);
}
```

**Estimated gain:** ~5-8% NPS improvement (SEE is called ~8-10 times per node for captures).

---

### 4. `makeMove`/`unmakeMove` copies `squares[8][8]` — 512 bytes per call

**Location:** `Board.cpp`  
**Cost:** `UndoInfo` contains a full `Piece squares[8][8]` snapshot = 64 × 8 bytes = 512 bytes. Copied on every `makeMove` and restored on every `unmakeMove`. At 30 chess moves per node: 30 × 512 = 15KB saved + 15KB restored = 30KB per node.

**Root cause:** The full squares snapshot was added as a safety net for edge cases (castling, en passant, promotion). The bitboard fields are also snapshotted separately.

**Fix requires restructuring:**  
Remove the `squares[8][8]` snapshot from `UndoInfo` and instead reconstruct `squares[][]` from the bitboard fields on unmake. The bitboards already capture all piece positions — `squares[][]` is redundant. This requires verifying that all edge cases (castling rook, EP pawn, promotion) are correctly handled by the bitboard restore path.

**Estimated gain:** ~8-12% NPS improvement.

---

### 5. `UndoInfo.pieceBBs[7]` — 56 bytes of bitboard snapshot per call

**Location:** `Board.cpp`  
**Cost:** 7 × 8 bytes = 56 bytes saved/restored per move. Minor but part of the overall `UndoInfo` bloat.

**Fix:** Only snapshot the bitboards that actually change (at most 2-3 per move). Use a delta-based undo for bitboards instead of a full snapshot.

**Estimated gain:** Negligible in isolation, meaningful combined with fix #4.

---

### 6. `forwardQ` L2 loop — uses float path, INT8 weights unused

**Location:** `DuckNNUE.cpp` — `forwardQ()`  
**Cost:** L2 has 128 outputs, each requiring a dot product over 1024 inputs (512 stm + 512 opp). Uses `L2_weights_T` (float, 8 values per AVX2 op). `L2_weights_T_q` (INT8) is quantized in `quantizeWeights()` but **never used in `forwardQ`** — the INT8 path is dead code.

**Fix (no restructuring):**  
Replace the float L2/L3 loops in `forwardQ` with `_mm256_maddubs_epi16` INT8 SIMD (32 values per op vs 8 for float = 4x throughput). The quantized weights and biases are already computed and stored.

**Estimated gain:** ~2-3x eval speed = significant NPS improvement since eval is called at every leaf.

---

### 7. Compound branching factor — structural limit

**Not fixable without algorithm change.**

At depth 2: ~30 chess moves × 7 duck placements = **210 leaf evals per node**.  
At depth 3: ~30 chess moves × 12 duck placements = **360 leaf evals per node**.

Standard chess at depth 3 has ~30 leaf evals per node (alpha-beta reduces ~30³ to ~30^1.5 ≈ 164). Duck chess cannot alpha-beta prune across duck placements the same way — the duck placement is the *second half* of a move, so you can't cut off the chess move based on duck placement scores.

**Possible approaches:**
- **Null duck move:** Evaluate the chess move with no duck placement as a lower bound. If this already beats beta, skip all duck placements. Cheap to implement, may give 10-20% node reduction.
- **Lazy duck evaluation:** Only evaluate the top 2-3 duck placements at depth 1 (currently 4). Already partially done with the `maxDucks` cap.
- **Separate duck search:** Treat duck placement as a separate 1-ply search after the chess move, with its own alpha-beta. This is a more fundamental restructuring of the search.

---

## Summary Table

| Item | Effort | Estimated Gain | Status |
|---|---|---|---|
| Pass acc as pointer (fix #1) | Medium — signature change | ~30-40% | Pending |
| Delta-undo postChess (fix #2) | Medium | ~10-15% | Pending |
| MVV-LVA for duck captures (fix #3) | Low — 5 lines | ~5-8% | Pending |
| Remove squares[][] from UndoInfo (fix #4) | High — needs verification | ~8-12% | Pending |
| INT8 L2/L3 in forwardQ (fix #6) | Low — check if active | ~2-3x eval | Needs investigation |
| Null duck move pruning (fix #7) | Medium | ~10-20% node reduction | Pending |
