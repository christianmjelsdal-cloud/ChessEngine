# Tier 1 Fixes — Implementation Summary

## Files Modified
- **Engine.h** — New member variables, TT prefetch helper
- **Engine.cpp** — Stack reduction, TT prefetch, staged move picking, fused accumulator
- **NNUE.h** — New `fusedCopyAndUpdateQ` declaration
- **NNUE.cpp** — New `fusedCopyAndUpdateQ` implementation

---

## Fix #1: Stack Overflow Prevention
**Problem:** `search()` allocated ~10-17 KB on the stack per recursion level:
- `MoveList moves` (256 × 20B = 5 KB)
- `Move quietsTriedArr[256]` (5 KB)
- `int scores[256]` inside `orderMoves()` (1 KB)

At MAX_PLY=64 plus qsearch depth, total stack usage approached 1 MB — dangerously close to Windows' default stack limit.

**Fix:**
- `quietsTriedArr` → **per-ply class member** `quietsTried_[MAX_PLY][64]` (~80 KB on heap, shared across all plies)
- `scores[]` → **class member** `mutable int orderScores_[MAX_MOVES]` (1 KB on heap)
- Added `MAX_QUIETS_TRACKED = 64` with bounds checking (64 quiet moves tracked is more than sufficient — history malus doesn't need hundreds)

**Stack savings:** ~6 KB per recursion level → search can now safely reach MAX_PLY without stack overflow.

---

## Fix #3: TT Prefetch
**Problem:** Transposition table entries are stored in a large hash table (96 MB). Each TT probe is an essentially random memory access that misses L1/L2 cache, costing 50-100+ cycles of stall time.

**Fix:**
- Added `_mm_prefetch(&tt_[hash % ttSize_], _MM_HINT_T0)` at the start of both `search()` and `qsearch()`
- In `search()`, the prefetch is issued before the repetition detection loop, giving ~50-100 cycles of useful work for the cache line to arrive
- In `qsearch()`, prefetched before the TT probe

**Expected gain:** +3-5% NPS (nodes per second). The prefetch runs in parallel with repetition detection, making the TT access essentially free.

---

## Fix #4: Staged Move Generation (Lazy Scoring + Selection Sort)
**Problem:** The engine generated, scored (including expensive SEE calls), and fully sorted ALL legal moves at every search node — even though ~90% of nodes cut off after testing just 1-3 moves.

**Fix:** Replaced `orderMoves(moves, board, ply, hashMove)` with a 3-stage approach:

1. **Hash move first** — Found in the legal move list and swapped to position 0. No scoring needed.
2. **Lazy scoring** — Remaining moves are scored only when the hash move doesn't cause a cutoff (line `if (i == 1 && !movesScored)`).
3. **Partial selection sort** — Instead of sorting all N moves upfront (O(N²)), picks the best from remaining moves one at a time (O(N) per pick). Since we usually only pick 1-3 moves before cutoff, total work is O(N) vs O(N²).

**Key insight:** At cut nodes (~90% of all nodes), the hash move causes a beta cutoff. With this fix, NO other moves are scored — eliminating all `scoreMove()` calls (which include expensive `see()` computations).

**Expected gain:** +10-25 Elo. The single largest algorithmic improvement in Tier 1.

---

## Fix #5: Fused Accumulator Copy+Update
**Problem:** The NNUE accumulator update was done in two steps:
1. Copy parent accumulator to child: `accStack_[ply+1] = accStack_[ply]` (4 KB memcpy — read parent, write child)
2. Increment child in-place: `incrementalUpdateQ(accStack_[ply+1], ...)` (read child, modify, write child)

Total: 3 full memory passes over 4 KB of data (read→write→read/write).

**Fix:** New `fusedCopyAndUpdateQ()` reads the parent accumulator, applies weight deltas (add/subtract), and writes directly to the child — all in a single AVX2 SIMD loop.

```
Before: read(parent) → write(child) → read(child) → modify → write(child)  [3 passes]
After:  read(parent) → apply delta → write(child)                           [2 passes]
```

**Implementation details:**
- Computes feature indices for removed piece (old square), captured piece, and added piece (new square)
- Loads parent values, subtracts/adds weight vectors, stores to child — all in one `for` loop
- King moves still trigger full `refreshAccumulatorQ()` (all features change)
- Special moves (castling, en passant, promotion) still use copy+refresh

**Expected gain:** +5-10% eval speed. Since NNUE evaluation is the hottest path in modern engines, this translates directly to higher NPS.

---

## What's NOT Changed (Intentional)
- **`orderMoves()` in qsearch** — Left as-is. Qsearch has fewer moves (captures only) and the sorting overhead is minimal.
- **`orderMoves()` in root search** — Called once per depth iteration, not a hot path.
- **`orderMoves()` in Duck Chess** — Separate search path, lower priority.
- **Null move accumulator copy** — Still uses plain copy (`accStack_[ply+1] = accStack_[ply]`). No piece movement means no weight delta to fuse.

---

## How to Apply
Replace these 4 files in your project:
- `Engine.h`
- `Engine.cpp`
- `NNUE.h`
- `NNUE.cpp`

All other source files are unchanged. The fixes are backward-compatible — no changes needed to Board, MoveGen, UCI, or training code.

## Fix #2: Windows Power Plan
This is an OS-level change, not a code change:
1. Open Windows Settings → System → Power & Sleep → Additional power settings
2. Select **High Performance** plan
3. Or run in an elevated PowerShell: `powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c`

Expected gain: +5-10% NPS (free — your CPU will boost higher and stay boosted).
