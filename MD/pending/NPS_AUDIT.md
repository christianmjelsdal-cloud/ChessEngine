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
| `forwardQ` L3: float AVX2 → INT8 `_mm256_maddubs_epi16` | ~5-10% eval |
| `isSquareAttacked`: manual ray-walk → magic bitboard lookups | ~5-10% std chess |
| `orderDuckPlacements`: full sort → partial selection sort (topN) | ~3-5% |
| Pass `parentAcc` as pointer to `searchDuck` — eliminates 420KB/node copy | ~30-40% |
| Delta-undo chess move on `parentAcc` — eliminates 60KB/node copy | ~10-15% |
| Remove `squares[8][8]` from `UndoInfo` — reconstruct from bitboards | ~8-12% |
| Null duck move pruning — skip duck placements when chess move beats beta | ~10-20% nodes |

---

## Remaining Bottlenecks

### 1. `TTEntry` is ~44 bytes — far above the 16-byte target

**Location:** `Engine.h` — `TTEntry` struct  
**Cost:** `TTEntry` contains `Move best{}` which includes `Square duckTo` (8 bytes for duck chess). Total size:
- `uint64_t key` = 8
- `int32_t score` = 4
- `int16_t depth` = 2
- `uint8_t flag` = 1
- `uint8_t gen` = 1
- `Move best` = `Square from`(8) + `Square to`(8) + `PieceType promotion`(1) + `Square duckTo`(8) = 25 + padding = ~28 bytes
- **Total: ~44 bytes** vs the 16-byte target in `TT_ENTRY_BYTES`

A 4M-entry TT at 44 bytes = **176MB** vs 64MB at 16 bytes. This causes massive cache pressure — TT entries span 3 cache lines instead of 1.

**Fix (medium effort):**  
Pack the move using `packMove()` (already implemented in `Types.h`) — stores from/to/promotion in 16 bits. For duck chess, store duck placement separately or omit it from TT (duck placement is re-searched anyway). Replace `Move best` with `uint16_t bestPacked`:
```cpp
struct TTEntry {
    uint64_t key   = 0;   // 8 bytes
    int32_t  score = 0;   // 4 bytes
    int16_t  depth = -1;  // 2 bytes
    uint8_t  flag  = 2;   // 1 byte
    uint8_t  gen   = 0;   // 1 byte
    uint16_t best  = 0;   // 2 bytes (packed move)
    // Total: 18 bytes → pad to 20 or pack to 16
};
```
With 16-byte entries: 4M entries = 64MB, fits in L3 cache on most CPUs. Cache miss rate drops dramatically.

**Estimated gain:** ~15-25% NPS improvement from better TT cache utilization.

---

### 2. `UndoInfo` in `Types.h` is dead code

**Location:** `Types.h` — bottom of file  
**Issue:** There are two `UndoInfo` structs: one in `Types.h` (never used) and `Board::UndoInfo` (used everywhere). The `Types.h` version is dead code that adds confusion and compile overhead.

**Fix (trivial):** Delete the `UndoInfo` struct from `Types.h`.

**Estimated gain:** None for NPS, but reduces confusion and compile time slightly.

---

### 3. `Board::UndoInfo` still snapshots all 7 `pieceBBs` — 56 bytes

**Location:** `Board.cpp` — `makeMove`  
**Cost:** 7 × 8 bytes = 56 bytes saved per move. With 30 chess moves per node: 30 × 56 = 1.68KB per node.

**Fix (medium effort):**  
Only snapshot the 2-3 bitboards that actually change per move. For a quiet move: only the moving piece type's bitboard and the color bitboards change. For a capture: also the captured piece type's bitboard. Track which bitboards changed and restore only those.

**Estimated gain:** ~3-5% NPS improvement.

---

### 4. `getLegalMoves` calls `makeMove`/`unmakeMove` for every pseudo-legal move

**Location:** `MoveGen.cpp` — `getLegalMoves`  
**Cost:** For standard chess, every pseudo-legal move is made and unmade to check legality. With ~30 pseudo-legal moves and `Board::UndoInfo` now at ~120 bytes (after removing squares[][]), this is 30 × 120 × 2 = 7.2KB per call.

**Note:** This only affects standard chess (duck chess skips legality checking). Not a duck chess bottleneck.

**Fix (high effort):**  
Use pin detection and check evasion to filter pseudo-legal moves without make/unmake. Standard approach in strong engines. Requires implementing pin bitboards and check detection via bitboard attacks.

**Estimated gain:** ~10-20% for standard chess NPS. No impact on duck chess.

---

### 5. `Engine` per-thread allocation in self-play — 12 × ~282KB

**Location:** `SelfPlayGen.cpp` — worker lambda  
**Cost:** Each of 12 worker threads allocates its own `Engine` on the heap. `Engine` contains:
- `history_[2][64][64]` = 65,536 ints = 256KB
- `contHist_[7][64][7][64]` = 200,704 ints = ~784KB
- `pvTable_[64][64]` = 4,096 Moves = ~114KB
- `accStack_[68]` = 68 × 2KB = 136KB
- `duckAccStack_[128]` = 128 × 2KB = 256KB
- TT: `SELFPLAY_TT_SIZE = 1<<20` entries × 44 bytes = **44MB per thread**

**Total per thread: ~45MB**. With 12 threads: **~540MB** just for Engine instances.

The TT dominates. At 44 bytes per entry and 1M entries, each thread's TT is 44MB. This is the primary memory pressure source.

**Fix:** Reduce `SELFPLAY_TT_SIZE` or fix TTEntry size (item #1 above). With 16-byte entries: 1M entries = 16MB per thread = 192MB total — much more manageable.

**Estimated gain:** Indirect — reducing memory pressure improves cache behavior across all threads.

---

### 6. `contHist_[7][64][7][64]` — 784KB per Engine instance

**Location:** `Engine.h`  
**Cost:** 7 × 64 × 7 × 64 × 4 bytes = 802,816 bytes ≈ 784KB. This is larger than L2 cache on most CPUs, meaning continuation history lookups frequently miss L2 and hit L3.

**Fix (low effort):**  
Reduce dimensions. The piece type dimension (7) includes `None` and `Duck` which are never used. Reduce to 6 (Pawn through King):
```cpp
int contHist_[6][64][6][64]{};  // 6×64×6×64×4 = 589,824 bytes ≈ 576KB
```
Or use a smaller table with a hash: `contHist_[piece_from_sq & 0x3FF][piece_to_sq & 0x3FF]` — 1024×1024×4 = 4MB (worse). Better: reduce to `[6][64][6][64]` and adjust indexing.

**Estimated gain:** ~2-5% from better cache utilization of history tables.

---

### 7. `shouldStop()` checks `nodes_ & 4095` — called every node

**Location:** `Engine.cpp` — `shouldStop()`  
**Cost:** Every node calls `shouldStop()` which checks `(nodes_ & 4095) == 0` and then calls `elapsedMs()` (a `std::chrono` call). The `std::chrono::steady_clock::now()` call is relatively expensive (~10-50ns). At 1600 NPS with 12 workers, this fires ~4 times per second per thread — negligible.

**Status:** Not a bottleneck at current NPS. Would matter at 100K+ NPS.

---

### 8. Duck placement TT — cache best duck square per position

**Location:** `Engine.cpp` — `searchDuck`  
**Concept:** After a chess move, the best duck placement for a given board position is deterministic (same position → same best duck). Cache it in the TT. If the same post-chess position is reached via different chess moves, reuse the cached duck placement score.

**Complexity:** Duck chess TT already stores the best chess move. Extending it to also store the best duck square requires either a larger TT entry or a separate duck TT.

**Estimated gain:** Unknown — depends on how often the same post-chess position is reached via different chess moves. Likely small at depth 2-3.

---

### 9. `MoveList` has `assert()` in hot path

**Location:** `Types.h` — `MoveList::operator[]`  
**Cost:** In Release builds, `assert()` is compiled out (`NDEBUG` defined). No cost.

**Status:** Not a bottleneck.

---

### 10. `Square` struct uses `int` fields — 8 bytes per square

**Location:** `Types.h`  
**Cost:** `Square` has `int rank` (4 bytes) + `int col` (4 bytes) = 8 bytes. `Move` has 3 squares = 24 bytes + 1 byte promotion = 25 bytes + padding. Using `int8_t` would reduce `Square` to 2 bytes and `Move` to 7 bytes.

**Fix (medium effort, high risk):**  
Change `Square::rank` and `Square::col` to `int8_t`. Use -1 as invalid sentinel (fits in int8_t). This would reduce `Move` from ~28 bytes to ~8 bytes, `TTEntry` from ~44 to ~20 bytes, `MoveList` from ~7KB to ~2KB.

Requires auditing all arithmetic on rank/col to ensure no overflow with int8_t.

**Estimated gain:** ~10-20% from reduced memory bandwidth across all data structures.

---

## Summary Table

| Item | Effort | Estimated Gain | Status |
|---|---|---|---|
| Pass acc as pointer — fix #1 | Medium | ~30-40% | ✅ Done |
| Delta-undo postChess — fix #2 | Medium | ~10-15% | ✅ Done |
| Remove squares[][] from UndoInfo — fix #3 | High | ~8-12% | ✅ Done |
| Magic bitboard `isSquareAttacked` — fix #5 | Low | ~5-10% std chess | ✅ Done |
| INT8 L2/L3 in forwardQ — fix #6 | Low | ~2-3x eval | ✅ Done |
| Partial sort for duck placements — fix #7 | Low | ~3-5% | ✅ Done |
| Null duck move pruning — fix #8a | Medium | ~10-20% nodes | ✅ Done |
| Pack `TTEntry.best` to uint16_t — new #1 | Medium | ~15-25% | ✅ Done |
| Delete dead `UndoInfo` from Types.h — new #2 | Trivial | cleanup | ✅ Done |
| Snapshot only changed `pieceBBs` — new #3 | Medium | ~3-5% | ✅ Done |
| Reduce `contHist_` dimensions — new #6 | Low | ~2-5% | ✅ Done |
| Reduce `Square` to int8_t — new #10 | Medium/High | ~10-20% | Pending |
| Duck placement TT — fix #8c | Medium | unknown | ✅ Done |
