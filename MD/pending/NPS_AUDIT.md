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
| `scoreMove` duck captures: SEE → MVV-LVA | ~5-8% |
| `forwardQ` L2: float AVX2 → INT8 `_mm256_maddubs_epi16` | ~2-3x eval |

---

## Remaining Bottlenecks

### 1. `*myAcc = postChess` — 420KB memcpy per node (HIGHEST PRIORITY)

**Location:** `Engine.cpp` — duck placement inner loop  
**Cost:** `QAccumulator` = 512 × 2 × int16 = 2KB. Copied once per duck placement per chess move.  
At depth 2: 7 ducks × 30 chess moves = 210 copies × 2KB = **420KB per node**.  
At depth 3: 12 ducks × 30 chess moves = 360 copies × 2KB = **720KB per node**.

**Root cause:** `searchDuck` at ply+1 reads `accStack[ply]` as its parent by index. The only way to communicate the post-duck accumulator to the child is to write it into `accStack[ply]` before recursing.

**Fix requires restructuring:**  
Add a `parentAcc` pointer parameter to `searchDuck`:
```cpp
int searchDuck(Board& board, int depth, int alpha, int beta, int ply,
               DuckNNUE::QAccumulator* accStack,
               DuckNNUE::QAccumulator* parentAcc);  // <-- add this
```
The duck loop applies the delta to `postChess` in-place and passes `&postChess` directly as `parentAcc`. The child uses `parentAcc` instead of `accStack[ply-1]`. Eliminates all 210 copies — only the 30 `postChess = *parentAcc` copies remain.

**Estimated gain:** ~30-40% NPS improvement.

---

### 2. `postChess = *parentAcc` — 60KB memcpy per node

**Location:** `Engine.cpp` — chess move loop  
**Cost:** 2KB × 30 chess moves = 60KB per node.

**Root cause:** Each chess move needs its own post-chess accumulator as a base for the duck delta loop. The parent must be preserved intact for the next chess move.

**Fix (medium effort):**  
Instead of copying, apply the chess delta to `parentAcc` in-place, use it for the duck loop, then undo the chess delta after. Turns 30 × 2KB copies into 30 × (2-4 `removeFeatureQ`/`addFeatureQ` calls). Each call is an AVX2 512-element loop (~32 ops) — much cheaper than a 2KB memcpy.

Requires careful handling of king moves (which trigger full refresh) and en passant.

**Estimated gain:** ~10-15% NPS improvement.

---

### 3. `makeMove` copies `squares[8][8]` — 512 bytes per call

**Location:** `Board.cpp` — `makeMove`/`unmakeMove`  
**Cost:** `UndoInfo` contains `Piece squares[8][8]` = 64 × 8 bytes = 512 bytes. Saved on every `makeMove`, restored on every `unmakeMove`. At 30 chess moves per node: 30 × 512 × 2 = **30KB per node** just for the squares snapshot.

**Root cause:** The full squares snapshot was added as a safety net. The bitboards are also snapshotted separately (redundant for piece positions).

**Fix requires restructuring:**  
Remove `squares[8][8]` from `UndoInfo`. Reconstruct `squares[][]` from the bitboard fields on unmake. The bitboards already capture all piece positions. Requires verifying all edge cases (castling rook, EP pawn, promotion) are correctly handled by the bitboard restore path.

**Estimated gain:** ~8-12% NPS improvement.

---

### 4. `makeMove` snapshots all 7 `pieceBBs` — 56 bytes per call

**Location:** `Board.cpp` — `makeMove`  
**Cost:** 7 × 8 bytes = 56 bytes saved/restored per move. Minor individually but part of `UndoInfo` bloat.

**Fix (medium effort):**  
Only snapshot the 2-3 bitboards that actually change per move (the moving piece type, captured piece type, and possibly promotion type). Use a delta-based undo.

**Estimated gain:** Negligible alone, meaningful combined with fix #3.

---

### 5. `isSquareAttacked` uses mailbox scan for sliders

**Location:** `Board.cpp`  
**Cost:** The slider check in `isSquareAttacked` uses a manual ray-walking loop (not magic bitboards). Called during standard chess legality checking — not in duck chess hot path, but affects standard chess NPS.

**Fix (low effort):**  
Replace the manual ray-walking with `BB::rookAttacks`/`BB::bishopAttacks` magic bitboard lookups. Already available in `Bitboard.h`.

```cpp
// Replace manual slider loops with:
Bitboard occ = occupied();
if (BB::rookAttacks(sq, occ) & (pieces(byColor, PieceType::Rook) | pieces(byColor, PieceType::Queen)))
    return true;
if (BB::bishopAttacks(sq, occ) & (pieces(byColor, PieceType::Bishop) | pieces(byColor, PieceType::Queen)))
    return true;
```

**Estimated gain:** ~5-10% for standard chess NPS. No impact on duck chess.

---

### 6. `forwardQ` L3 still uses float path

**Location:** `DuckNNUE.cpp` — `forwardQ()`  
**Cost:** L3 has 64 outputs over 128 inputs. Currently uses float AVX2 (8 values per op). L2 output (`l2Out`) is float, so quantizing it to uint8 for INT8 L3 requires an extra conversion step.

**Fix (low effort):**  
Quantize `l2Out` to uint8 inline (clamp to [0,127]) and use `_mm256_maddubs_epi16` for L3 as well. L3_SIZE=64 so the loop is only 8 iterations of 16 int8 values — the conversion overhead is minimal.

**Estimated gain:** ~5-10% eval speed (L3 is smaller than L2 so less impact).

---

### 7. `getDuckPlacements` returns all empty squares (~60)

**Location:** `MoveGen.cpp`  
**Cost:** Returns all empty squares via bitboard popcount — fast. But `orderDuckPlacements` then scores all 60 squares even though only `maxDucks` (4-12) will be used.

**Fix (low effort):**  
Score only the top `maxDucks` squares using a partial insertion sort (stop after finding the top N). Avoids scoring ~50 squares that will never be used.

**Estimated gain:** ~3-5% (scoring is cheap after the bitboard fix, but still 60 calls vs 7-12).

---

### 8. Compound branching factor — structural limit

**Not fixable without algorithm change.**

At depth 2: ~30 chess moves × 7 duck placements = **210 leaf evals per node**.  
At depth 3: ~30 chess moves × 12 duck placements = **360 leaf evals per node**.

Standard chess at depth 3 has ~30 leaf evals per node. Duck chess cannot alpha-beta prune across duck placements — the duck placement is the second half of a move, so you can't cut off the chess move based on duck placement scores.

**Possible approaches:**

**8a. Null duck move pruning (medium effort, ~10-20% node reduction)**  
Before evaluating any duck placements for a chess move, evaluate the position with no duck placement (or duck staying put) as a quick lower bound. If this already beats beta, skip all duck placements for this chess move.

**8b. Separate duck search (high effort)**  
Treat duck placement as a separate 1-ply maximization after the chess move, with its own alpha-beta. The chess move loop becomes a standard alpha-beta, and the duck placement is a separate inner loop that finds the best duck square for the current chess move. This is a more fundamental restructuring but could reduce the effective branching factor significantly.

**8c. Duck placement TT (medium effort)**  
Cache the best duck placement for a given board position in the TT. If the same post-chess position is reached via different chess moves, reuse the cached duck placement score.

---

### 9. `MoveList` and `SquareList` — fixed-size stack arrays

**Location:** `Types.h` (presumably)  
**Cost:** `MoveList` is a fixed-size array (256 moves). Each `getDuckChessMoves` call fills it from scratch. No issue with allocation, but the array is 256 × sizeof(Move) bytes on the stack — potentially cache-unfriendly if large.

**Status:** Likely fine as-is. Worth checking sizeof(Move) to ensure it's compact.

---

## Summary Table

| Item | Effort | Estimated Gain | Status |
|---|---|---|---|
| Pass acc as pointer — fix #1 | Medium | ~30-40% | ✅ Done |
| Delta-undo postChess — fix #2 | Medium | ~10-15% | ✅ Done |
| Remove squares[][] from UndoInfo — fix #3 | High | ~8-12% | ✅ Done |
| Magic bitboard `isSquareAttacked` — fix #5 | Low | ~5-10% std chess | ✅ Done |
| INT8 L3 in forwardQ — fix #6 | Low | ~5-10% eval | ✅ Done |
| Partial sort for duck placements — fix #7 | Low | ~3-5% | ✅ Done |
| Null duck move pruning — fix #8a | Medium | ~10-20% nodes | ✅ Done |
| Duck placement TT — fix #8c | Medium | unknown | Pending |
| MVV-LVA for duck captures — fix #3 prev | Done | ~5-8% | ✅ Done |
| INT8 L2 in forwardQ — fix #6 prev | Done | ~2-3x eval | ✅ Done |
