# Chess Engine Full Audit Report

> **Generated:** Tuesday, 24 March 2026  
> **Codebase:** ~25,500 lines (C++17, Python, PowerShell)  
> **Files audited:** Engine.cpp, Board.cpp, MoveGen.cpp, Bitboard.cpp, NNUE.cpp, NNUETrainer.cpp, TrainingRunner.cpp, SelfPlayGen.cpp, UCI.cpp, GameLogic.cpp, VisualGame.cpp, main.cpp, Test.cpp, SmokeTest.cpp, train_nnue.py, pgn_to_training.py, training_format.py, pipeline.ps1, smart_push.ps1, elo_calibration.ps1, pgo_build.sh, train.bat, build_tests.bat, CMakeLists.txt

---

## Executive Summary

This codebase implements a full-featured chess engine with NNUE (Efficiently Updatable Neural Network) evaluation, incremental NNUE training from self-play data, a UCI protocol interface, a graphical chess GUI, and a multi-stage automated training pipeline. The engine supports standard chess, duck chess, and multi-PV search, and includes perft and smoke-test harnesses for validation. The overall architecture is sophisticated and the majority of the code reflects experienced authorship — bitboard move generation is bitboard-native, the NNUE implementation includes quantized inference with AVX2 acceleration, and the search implements most modern alpha-beta enhancements (LMR, NMP, singular extensions, history heuristics, etc.).

However, the audit uncovered **92 findings** across all components, including one **Critical** correctness defect (TT mate-score ply normalization) and fourteen **High** severity issues covering search correctness, NNUE evaluation, UCI thread safety, and training pipeline reliability. These are not cosmetic issues — the critical finding can cause the engine to prefer longer mating sequences or miss shorter ones, and several high-severity findings cause silent evaluation corruption or race conditions in production.

The most important findings to address are: (1) the transposition-table mate-score ply adjustment bug, which corrupts mate distance universally; (2) the NNUE incremental update functions that silently produce wrong evaluations for en passant and promotion without caller-contract enforcement; (3) platform portability failures (`_aligned_malloc` is MSVC-only and will not compile on Linux/GCC); (4) the `evaluateMovesNNUE` score-perspective inversion in SelfPlayGen, which corrupts move-quality data fed into NNUE training; and (5) the lack of runtime AVX2 detection, which will crash silently on older CPUs.

**Recommended priority order:** Fix Critical and High issues first (engine correctness and training data quality). Address Medium issues in the next pass (thread safety, portability, code clarity). Low and Info findings can be addressed opportunistically during normal development.

---

## Summary Table

| Severity | Section 1: Core | Section 2: NNUE | Section 3: UCI/GUI/Tests | Section 4: Scripts/Build | Total |
|----------|:-:|:-:|:-:|:-:|:-:|
| **Critical** | 1 | 0 | 0 | 0 | **1** |
| **High** | 3 | 6 | 2 | 4 | **15** |
| **Medium** | 9 | 10 | 12 | 11 | **42** |
| **Low** | 5 | 5 | 9 | 5 | **24** |
| **Info** | 2 | 0 | 3 | 0 | **5** |
| **Subtotal** | **20** | **21** | **26** | **20** | **92** |

---

## Section 1: Core Engine

**Files:** Engine.cpp, Board.cpp, MoveGen.cpp, Bitboard.cpp

---

### 1.1 TT Mate Scores Not Adjusted for Ply Distance

**Severity:** 🔴 Critical  
**File:** Engine.cpp  
**Lines:** 1126–1128, 1043, 1059, 965–967, 1533

**Description:**  
When storing scores in the transposition table (line 1533: `tte.score = (int16_t)bestScore`), mate scores of the form `(MATE_SCORE - ply)` or `-(MATE_SCORE - ply)` encode the current ply distance from the root. When the same position is later retrieved from the TT at a *different* ply depth, the raw score is used directly (lines 1126–1128) without re-normalising for the ply difference.

Concretely: a "mate in 5 from ply 3" is stored as `MATE_SCORE - 8`. When retrieved at ply 1, it is interpreted as "mate in 8 from root" instead of the correct "mate in 6 from ply 1". This corrupts mate-distance reporting across the entire tree and can cause the engine to prefer longer mating sequences or miss shorter ones. The same bug exists in the qsearch TT path (store at lines 1043 and 1059; retrieval at lines 965–967).

**Suggested Fix:**  
When *storing* in TT, convert ply-relative mate scores to root-relative:
```cpp
// Store
int16_t ttScore = (int16_t)bestScore;
if (ttScore > MATE_SCORE - 100) ttScore += ply;    // "mate in N" → root-relative
if (ttScore < -MATE_SCORE + 100) ttScore -= ply;   // "mated in N" → root-relative
tte.score = ttScore;

// Retrieve
int score = tte.score;
if (score > MATE_SCORE - 100) score -= ply;   // root-relative → current ply
if (score < -MATE_SCORE + 100) score += ply;
```
Apply this correction in both `search()` and `qsearch()` TT store/probe paths.

---

### 1.2 TT Best Move Preserved from Fail-Low on Fail-High Overwrite

**Severity:** 🟠 High  
**File:** Engine.cpp  
**Lines:** 1524–1530

**Description:**  
When an existing TT entry has flag `TT_EXACT` or `TT_UPPER` and the new entry is `TT_LOWER` (fail-high), lines 1527–1531 preserve the *old* entry's best move instead of storing the move that caused the beta cutoff. The intent appears to be that the fail-high move is "not necessarily best," but this is backwards: the fail-high (cutoff) move is precisely the most valuable move to try first for move ordering. The old entry's move (from a `TT_UPPER`/fail-low node) is the last move searched in that prior pass and has poor ordering value. This degrades move ordering quality at all TT-hit nodes.

**Suggested Fix:**  
Remove the preservation logic at lines 1527–1531 and always store the cutoff move for `TT_LOWER` entries:
```cpp
tte.bestMove = packMove(bestMove);
```

---

### 1.3 Null Move Overwrites `searchStack_[ply]`, Breaking Repetition Detection

**Severity:** 🟠 High  
**File:** Engine.cpp  
**Lines:** 1184, 1196

**Description:**  
During null move pruning, line 1184 stores the null-move hash at `searchStack_[ply]`. Line 1196 restores the real hash afterward. However, *between* these two lines, the null-move child search at `ply+1` scans `searchStack_[0..ply]` for repetitions using the null-move hash for position `[ply]`. Two failure modes result:

1. If the null-move hash coincidentally matches a position in the search stack, a false repetition draw is returned.
2. The *real* hash at this ply is temporarily hidden, so a genuine three-fold repetition involving this position may be missed.

**Suggested Fix:**  
Do not overwrite `searchStack_[ply]` for the null move. Store the null-move hash at `searchStack_[ply+1]` (or a dedicated slot) so the child search's repetition scan at `0..ply` always sees the real hashes.

---

### 1.4 Repetition Detection Scans Every Ply Instead of Stepping by 2

**Severity:** 🟠 High  
**File:** Engine.cpp  
**Lines:** 1100–1112

**Description:**  
The repetition scan at lines 1100–1101 checks all plies from `0` to `ply-1`. A position can only repeat with the *same side to move*, which requires an even number of plies between occurrences. Scanning every ply wastes half the iterations and can produce false positives if two positions with *different* sides to move have colliding Zobrist hashes (an unlikely but possible event given imperfect hashing). The game-history scan (lines 1106–1111) has the same problem.

**Suggested Fix:**  
```cpp
// Search stack: step by 2
for (int i = ply - 2; i >= 0; i -= 2) { ... }

// Game history: step by 2
for (int i = (int)gameHistory_.size() - 2; i >= 0; i -= 2) { ... }
```
This halves the work and eliminates false same-hash/different-side matches.

---

### 1.5 Singular Extension Search Clobbers `pvLength_[ply]`

**Severity:** 🟡 Medium  
**File:** Engine.cpp  
**Lines:** 1251, 1250, 1252

**Description:**  
The singular extension verification search at line 1251 calls `search()` at the *same* ply (not `ply+1`), which causes it to write to `pvLength_[ply]`. While `searchStack_[ply]` is saved and restored around the call, `pvLength_[ply]` is overwritten by the singular search and **not** restored. This corrupts the PV for the main search at this ply, potentially producing garbage PV output in info strings or incorrect ponder moves.

**Suggested Fix:**  
Save and restore `pvLength_[ply]` around the singular extension call, mirroring the `searchStack_[ply]` save/restore pattern:
```cpp
int savedPvLength = pvLength_[ply];
// ... singular search ...
pvLength_[ply] = savedPvLength;
```
Alternatively, call the singular search at `ply+1` to avoid clobbering the parent's state entirely.

---

### 1.6 Countermove Lookup Uses Zero-Initialised `previousMoves_` (a1→a1 Phantom Move)

**Severity:** 🟡 Medium  
**File:** Engine.cpp  
**Lines:** 886–891, 1845

**Description:**  
`previousMoves_` is `memset` to 0 at line 1845, making `from = {0, 0}` (a1). The `isValid()` check at line 886 returns `true` for `(0, 0)` since a1 is a valid square. At nodes where `previousMoves_[ply-1]` was never set (e.g., after null-move children or at early plies in helper threads), the code uses the phantom a1→a1 move as a countermove key. This silently pollutes both the countermove table and the countermove history table with entries keyed to a non-existent move, degrading move ordering.

**Suggested Fix:**  
Initialise `previousMoves_` with a sentinel where `from = {-1, -1}` so `isValid()` returns `false`. Alternatively, add a dedicated `Move::none()` constant checked explicitly:
```cpp
memset(previousMoves_, 0xFF, sizeof(previousMoves_)); // -1 in all bytes
```

---

### 1.7 LMP Threshold Integer Division Loses Precision When Not Improving

**Severity:** 🟡 Medium  
**File:** Engine.cpp  
**Lines:** 1355

**Description:**  
Line 1355: `LMP_THRESHOLD[depth] / (improving ? 1 : 2)`. When `improving` is false and the threshold is odd (e.g., `LMP_THRESHOLD[1] = 5`), integer division gives `5/2 = 2` instead of the intended `2.5`, rounded in the unsafe truncation direction. Given `LMP_THRESHOLD = {0, 5, 8, 13}`:
- Depth 1, not improving: allows only **2** quiet moves instead of ~2–3
- Depth 3, not improving: allows only **6** quiet moves instead of ~6–7

This makes LMP incorrectly aggressive when not improving at odd-threshold depths.

**Suggested Fix:**  
Use ceiling division for the not-improving case:
```cpp
int lmpThresh = improving ? LMP_THRESHOLD[depth] : (LMP_THRESHOLD[depth] + 1) / 2;
```
Or maintain a separate `LMP_THRESHOLD_NOT_IMPROVING[]` table.

---

### 1.8 `applyMove` Does Not Update `nonPawnMaterial`, `phase`, or Hash

**Severity:** 🟡 Medium  
**File:** Board.cpp  
**Lines:** 2257–2368

**Description:**  
`Board::applyMove()` (used in duck chess search at Engine.cpp line 1634 and PV validation at line 2063) does not update `nonPawnMaterial[]`, `phase`, or the incremental Zobrist hash. Currently, PV validation only checks legality so the issue is latent. However, if `applyMove` is ever called in a path that subsequently queries material balance, phase interpolation, or requires a correct hash (e.g., repetition detection after PV validation), the engine will silently use stale values.

**Suggested Fix:**  
Either document explicitly that `applyMove` is only for duck chess and PV legality validation (add an assertion), or add the missing `nonPawnMaterial`/`phase`/hash updates to match `makeMove`'s behaviour. Consider deprecating `applyMove` in favour of `makeMove` + `UndoInfo`.

---

### 1.9 TT Generation Skip in `clearSearchState` May Produce Unpredictable Wrapping

**Severity:** 🟡 Medium  
**File:** Engine.cpp  
**Lines:** 350, 1692, 1523

**Description:**  
Line 350: `ttGen_ = (ttGen_ + TT_MAX_AGE + 1) & 63` attempts to advance the generation past all valid ages. If `getBestMove` also increments `ttGen_` by 1 on each call (line 1692), the interaction of the two incrementors can cause unpredictable generation wrapping. For example, if `TT_MAX_AGE = 3`, `clearSearchState` bumps by 4. If `getBestMove` runs 60+ iterations, generations wrap around the 6-bit counter and the "staleness" criterion in the replacement logic (line 1523) may treat new entries as old, causing premature TT eviction.

**Suggested Fix:**  
Use a single consistent TT-aging mechanism. One clean approach: `clearSearchState` sets a `ttCleared_` flag; the next `getBestMove` call resets `ttGen_` to 0 unconditionally, after which `getBestMove` increments per-iteration as normal.

---

### 1.10 Handcrafted Eval Iterates All 64 Squares via Mailbox Instead of Bitboards

**Severity:** 🟡 Medium  
**File:** Engine.cpp  
**Lines:** 606–667

**Description:**  
The `evaluate()` function (used when NNUE is unavailable) iterates all 64 mailbox squares to accumulate PST scores, phase, mobility, and pawn structure. Since bitboards are already maintained, piece iteration could use `popLsb` on `colorBB & pieceBB`, skipping all empty squares. In a typical middlegame position with ~32 pieces, this wastes ~32 iterations per `evaluate()` call. The inner `countMobilityBB` already uses bitboards — the outer loop does not.

**Suggested Fix:**  
```cpp
for (int pt = 0; pt < NUM_PIECE_TYPES; pt++) {
    Bitboard bb = board.colorBB[color] & board.pieceBB[pt];
    while (bb) {
        int sq = popLsb(bb);
        // accumulate PST, mobility, etc.
    }
}
```

---

### 1.11 `searchDuck` Creates Full `Board` Copy Per Chess Move

**Severity:** 🟡 Medium  
**File:** Engine.cpp  
**Lines:** 1633–1677

**Description:**  
Line 1633: `Board temp = board;` creates a full copy of the `Board` struct for *every* chess move in the duck search. With ~30 legal moves per position and recursive depth, this creates thousands of large object copies. The inner duck-placement loop already uses in-place save/restore (lines 1648–1662), but the outer chess-move level still copies the full board. The standard chess search avoids this entirely via `makeMove`/`unmakeMove`.

**Suggested Fix:**  
Use `makeMove`/`unmakeMove` for the chess-move leg of `searchDuck`, mirroring the standard search. Only save/restore the minimal necessary state rather than copying the entire `Board` struct.

---

### 1.12 `isKingCaptured` Scans All 64 Squares Instead of Using Bitboard

**Severity:** 🟡 Medium  
**File:** MoveGen.cpp  
**Lines:** 3378–3385

**Description:**  
`MoveGen::isKingCaptured` iterates all 64 mailbox squares to detect whether a king of the given colour is present. This is called frequently in duck chess search. A bitboard lookup is O(1).

**Suggested Fix:**  
```cpp
bool isKingCaptured(const Board& board, Color color) {
    return !board.pieces(color, PieceType::King);
}
```

---

### 1.13 Qsearch TT Prefetch Too Close to Access to Have Effect

**Severity:** 🟡 Medium  
**File:** Engine.cpp  
**Lines:** 953, 957–958

**Description:**  
Line 953 prefetches `tt_[board.hash % ttSize_]`. Lines 957–958 immediately compute the same index and load the same cache line. Only ~5 instructions separate the prefetch from the access, which is far too few for the prefetch to complete (typical memory latency is 100–300 cycles). The prefetch is providing no measurable benefit.

**Suggested Fix:**  
Move the prefetch to the top of the qsearch function (before the ply-limit check), or remove it if there is insufficient work between the prefetch and the TT probe to justify it.

---

### 1.14 `orderDuckPlacements` Off-by-One Potential with Empty List

**Severity:** 🟢 Low  
**File:** Engine.cpp  
**Lines:** 1591

**Description:**  
Line 1591: `for (int i = 0; i < placements.count - 1; i++)`. If `placements.count` is an unsigned type and equals 0, the subtraction `0 - 1` wraps to a very large positive number, causing the loop to run billions of iterations and access out-of-bounds memory.

**Suggested Fix:**  
Guard with an early return or use a safe comparison:
```cpp
if (placements.count <= 1) return;
// or:
for (int i = 0; i + 1 < (int)placements.count; i++)
```

---

### 1.15 `searchDuck` Uses `goto` for Nested Loop Control

**Severity:** 🟢 Low  
**File:** Engine.cpp  
**Lines:** 1668–1673

**Description:**  
Line 1668 uses `goto nextChessMove;` to break out of the inner duck-placement loop on beta cutoff. While functionally correct, `goto` complicates control-flow analysis, and a future edit (e.g., adding cleanup code before `nextChessMove`) could inadvertently skip the cleanup.

**Suggested Fix:**  
Replace with a flag variable:
```cpp
bool cutoff = false;
for (auto& duck : placements) {
    // ...
    if (alpha >= beta) { cutoff = true; break; }
}
if (cutoff) break; // break outer loop
```
Or use a lambda for the inner search that returns a `bool cutoff` value.

---

### 1.16 `removeBitboard` Uses XOR Which Corrupts State If Piece Absent

**Severity:** 🟢 Low  
**File:** Board.cpp  
**Lines:** 2936–2940

**Description:**  
`removeBitboard` XORs the bit for the piece square out of `pieceBB[pt]`. If the bit was not already set (piece not actually present), XOR will *set* the bit, silently corrupting the bitboard. There is no assertion or guard. A single bug in `makeMove`/`unmakeMove` could desynchronise all bitboards without any error.

**Suggested Fix:**  
Add a debug-mode assertion:
```cpp
assert((pieceBB[pt] & squareBB(sq)) != 0
    && "removeBitboard: piece not present in bitboard");
```

---

### 1.17 Passed Pawn Detection Uses O(n²) Mailbox Scan Per Pawn

**Severity:** 🟢 Low  
**File:** Engine.cpp  
**Lines:** 685–721

**Description:**  
The passed pawn evaluation iterates all 64 squares for each pawn, scanning forward through multiple ranks and files. With bitboards available, passed pawn detection is O(1) per pawn using a precomputed mask: a pawn is passed if no opponent pawns exist on the same or adjacent files ahead of it.

**Suggested Fix:**  
```cpp
// Precompute once (indexed by square and color)
Bitboard passedMask[2][64]; // fill at startup

// In evaluate():
bool passed = !(board.pieces(opponent, PieceType::Pawn) & passedMask[color][sq]);
```

---

### 1.18 Stack-Allocated `excludedRootMoves` Array Has Fixed Size 500

**Severity:** 🟢 Low  
**File:** Engine.cpp  
**Lines:** 1914, 2046

**Description:**  
Line 1914: `Move excludedRootMoves[500]` is stack-allocated with no bounds check at line 2046 when adding to it. While `multiPV` is typically small (1–10), there is no enforcement. An unusually high `multiPV` setting (e.g., via a malformed UCI command) could overflow the stack array.

**Suggested Fix:**  
```cpp
if (numExcluded < 500) excludedRootMoves[numExcluded++] = bestMoveIter;
```
Or size the array to `effectiveMultiPV`, which is bounded at configuration time.

---

### 1.19 Null Move Returns `beta` (Fail-Hard) Instead of `nullScore` (Fail-Soft)

**Severity:** ℹ️ Info  
**File:** Engine.cpp  
**Lines:** 1199–1200

**Description:**  
Line 1200: `return beta` discards the actual null-move score. In a fail-soft framework, returning `nullScore` preserves more information for the parent node's TT entry. Returning `beta` causes the parent to store a less accurate bound, which can reduce the quality of the TT entry used in subsequent iterations.

**Suggested Fix:**  
Consider `return nullScore` (or `return std::max(nullScore, beta)` to avoid returning less than beta). Many engines cap at beta: `return nullScore >= beta ? beta : nullScore;`.

---

### 1.20 `drawScore` Contempt Sign May Be Inverted for Side to Move

**Severity:** ℹ️ Info  
**File:** Engine.cpp  
**Lines:** 25–31

**Description:**  
`drawScore()` returns a bias based on `rootEval_` and `contemptCp_`. The formula `bias = -rootEval_ / 4 - contemptCp_` makes draws unfavourable when the engine is winning (from the root side's perspective). However, `drawScore` is returned at arbitrary plies where the side to move alternates. When the *opponent* is the one being offered a draw, the sign should be negated — a draw being bad for us is *good* for the opponent.

**Suggested Fix:**  
Make `drawScore` ply-aware by negating the bias on odd plies:
```cpp
int bias = -rootEval_ / 4 - contemptCp_;
return (ply % 2 == 0) ? bias : -bias;
```

---

## Section 2: NNUE & Training

**Files:** NNUE.cpp, NNUETrainer.cpp, TrainingRunner.cpp, SelfPlayGen.cpp

---

### 2.1 `incrementalUpdate` Does Not Handle En Passant or Promotions

**Severity:** 🟠 High  
**File:** NNUE.cpp  
**Lines:** 171–231

**Description:**  
The comment at lines 171–175 explicitly acknowledges that `incrementalUpdate()` does **not** handle en passant (the captured pawn is on a different square than `toSq`) or promotions (adds `Pawn` at `toSq` instead of the promoted piece type). The function always adds `movedPiece` at `toSq`. The code relies entirely on callers in Engine.cpp invoking `refreshAccumulator` for these special moves.

If this caller contract is ever broken (e.g., by a future refactor, new call site, or the contract being silently violated by a bug in the caller), evaluations for en passant and promotion positions will be silently wrong with no diagnostic. Promotion bugs are especially damaging because the wrong piece feature is added and the pawn feature is never removed.

**Suggested Fix:**  
Add explicit assertions (enabled in debug builds) to catch contract violations:
```cpp
// In incrementalUpdate(), guard promotions:
assert(!(toRank == 0 || toRank == 7) || movedPiece.type != PieceType::Pawn,
    "incrementalUpdate called for promotion — use refreshAccumulator");

// Guard en passant:
assert(!isEnPassant, "incrementalUpdate called for en passant — use refreshAccumulator");
```
Long-term, consider accepting an `isEnPassant` flag and `promotedPieceType` parameter to make the function capable of handling these cases safely.

---

### 2.2 `incrementalUpdateQ` Has the Same En Passant/Promotion Gap

**Severity:** 🟠 High  
**File:** NNUE.cpp  
**Lines:** 845–898

**Description:**  
The quantised incremental update `incrementalUpdateQ()` mirrors the float version's limitation: no en passant or promotion handling. Same silent-corruption risk if caller contract is violated.

**Suggested Fix:**  
Identical to finding 2.1 — add debug assertions for promotion and en passant paths, and consider adding first-class handling for both cases.

---

### 2.3 `SimpleSearcher` Uses Full Board Copies and Full NNUE Refresh Per Node

**Severity:** 🟠 High  
**File:** NNUETrainer.cpp  
**Lines:** 1220–1254

**Description:**  
The `SimpleSearcher` used for ELO estimation copies the `Board` struct for every move in both alpha-beta search and quiescence, then calls `net->evaluate()` which performs a full `refreshAccumulator` + forward pass each time. For alpha-beta at depth 4+ with quiescence, this means thousands of full NNUE evaluations per move decision. The main engine's search uses incremental accumulator updates (via `makeMove`/`unmakeMove` with an accumulator stack) which are orders of magnitude faster.

**Suggested Fix:**  
Use incremental accumulator updates (`makeMove`/`unmakeMove` with a `QAccumulator` stack) instead of `Board` copies + full refresh in `SimpleSearcher`. Alternatively, for ELO estimation, reuse the actual `Engine` class for both sides (engine vs engine), which already has all the performance optimisations.

---

### 2.4 `SimpleSearcher::getBestMove` Searches with Full-Window Alpha-Beta at Every Depth

**Severity:** 🟠 High  
**File:** NNUETrainer.cpp  
**Lines:** 1302–1347

**Description:**  
The iterative deepening root search at line 1324 calls `alphaBeta(child, depth-1, -MATE_SCORE, MATE_SCORE, 1)` for every root move at every iteration. This uses no aspiration windows, so each iteration redundantly re-searches with full window from scratch. Combined with the full-board-copy + full-NNUE-refresh problem (finding 2.3), ELO estimation is extremely slow.

**Suggested Fix:**  
Add aspiration windows around the previous iteration's score (e.g., ±50 centipawns initial window, widening on fail). Alternatively, reuse the main `Engine` class for the NNUE side in ELO estimation so all existing search optimisations apply automatically.

---

### 2.5 `_aligned_malloc` / `_aligned_free` Are MSVC-Only

**Severity:** 🟠 High  
**File:** NNUETrainer.cpp  
**Lines:** 2075–2078

**Description:**  
The code uses `_aligned_malloc` and `_aligned_free` (MSVC CRT extensions). On GCC/Clang/Linux these functions do not exist and the code will fail to compile. The `AlignedDeleter` struct also calls `_aligned_free`. This makes the training code non-portable and blocks Linux/CI builds.

**Suggested Fix:**  
Use C++17 portable aligned allocation:
```cpp
// Allocate
void* ptr = std::aligned_alloc(alignment, size);
// Deallocate  
std::free(ptr);
```
Or use `alignas(32)` with `std::vector` and a custom aligned allocator. Wrap the platform difference in a utility header:
```cpp
#ifdef _MSC_VER
  #define ALIGNED_ALLOC(align, size) _aligned_malloc(size, align)
  #define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
  #define ALIGNED_ALLOC(align, size) std::aligned_alloc(align, size)
  #define ALIGNED_FREE(ptr) std::free(ptr)
#endif
```

---

### 2.6 `evaluateMovesNNUE` Score Perspective May Be Inverted

**Severity:** 🟠 High  
**File:** SelfPlayGen.cpp  
**Lines:** 7989–7990

**Description:**  
`evaluateMovesNNUE` computes `ev` from the post-move board's perspective: `board.turn` is the *opponent* after `makeMove`, so `forwardQ` returns positive when the opponent is doing well. The code then does:
```cpp
scores[i] = (stm == White) ? ev : -ev;
```
where `stm` is the side that *was* to move. If `stm = White`, the opponent is Black, and `ev` is from Black's (opponent's) perspective. For the side choosing the move (White), we want `-ev`, not `ev`. This formula appears inverted — it returns the opponent's evaluation as the score for the moving side.

**Suggested Fix:**  
Negate unconditionally to convert from opponent-perspective to mover-perspective:
```cpp
scores[i] = -ev;
```
Verify against the NNUE evaluation convention (positive = good for the side whose turn it is) and add unit tests confirming polarity.

---

### 2.7 STM/OPP Gradient Mapping Variable Naming Is Confusing and Error-Prone

**Severity:** 🟡 Medium  
**File:** NNUETrainer.cpp  
**Lines:** 2421–2426

**Description:**  
When `sideToMove` is Black, the STM accumulator is `blackAcc` and OPP is `whiteAcc`. The gradient-routing logic at lines 2423–2424 correctly maps `dLdL1Pre` halves to white/black accumulators, but the variable names `stmGrad`/`oppGrad` refer to *perspective* (whose accumulator), not to the layout in `dLdL1Pre`. The comment `AUDIT FIX C2` confirms this was previously analysed, but the variable naming remains confusing and is a maintenance hazard for future gradient changes.

**Suggested Fix:**  
Rename the variables to make the mapping explicit:
```cpp
float* dLdL1Pre_white = isWhite ? dLdL1Pre : &dLdL1Pre[L1_SIZE];
float* dLdL1Pre_black = isWhite ? &dLdL1Pre[L1_SIZE] : dLdL1Pre;
```
And add a comment explaining that `dLdL1Pre[0..L1_SIZE-1]` always corresponds to the STM half and `dLdL1Pre[L1_SIZE..]` to the OPP half.

---

### 2.8 `Opening Moves in `playGame` Use `makeMove` with Discarded `UndoInfo`

**Severity:** 🟡 Medium  
**File:** SelfPlayGen.cpp  
**Lines:** 8113–8118, 8201, 8228, 8233, 8236

**Description:**  
In the opening and post-opening phases, moves are played using:
```cpp
{ UndoInfo undo; board.makeMove(legalMoves[dist(rng)], undo); }
```
The `UndoInfo` is created on the stack and immediately discarded when the block closes. This is functionally correct (the moves are intentionally permanent), but the pattern looks like an accident — it appears that `undo` might be needed. This confuses future readers and could lead to incorrect "fixes" that break the game loop.

**Suggested Fix:**  
Replace with `board.applyMove()` to make intent explicit, or add a comment:
```cpp
{ UndoInfo undo_unused; board.makeMove(move, undo_unused); } // undo not needed; moves are permanent
```

---

### 2.9 `fusedCopyAndUpdateQ` API Is Confusing About Pre- vs Post-Move Board

**Severity:** 🟡 Medium  
**File:** SelfPlayGen.cpp  
**Lines:** 7977–7983

**Description:**  
In `evaluateMovesNNUE()`, `fusedCopyAndUpdateQ` is called with the post-move `board` and pre-move piece info (`movedPiece.type/color`, `capturedPce`). The `board` parameter is only used in the king-move full-refresh path; for non-king moves, the pre-move piece info drives the delta computation. This dual-use API is undocumented and confusing.

**Suggested Fix:**  
Add a clear doc comment to `fusedCopyAndUpdateQ` explaining: "The `board` parameter is only used for king-move full refresh (post-move board state). For non-king moves, the `movedPiece`/`capturedPce` parameters drive the delta computation."

---

### 2.10 `thread_local` 544 KB Weight Buffers in Training Loop

**Severity:** 🟡 Medium  
**File:** NNUETrainer.cpp  
**Lines:** 2098–2099

**Description:**  
The training loop declares `thread_local alignas(32) float L2_weights_T[L2_SIZE][L1_SIZE * 2]` and `L3_weights_T[L3_SIZE][L2_SIZE]`. With `L1_SIZE=512`, `L2_SIZE=128`, `L3_SIZE=64`:
- `L2_weights_T`: 128 × 1024 × 4 = **512 KB**
- `L3_weights_T`: 64 × 128 × 4 = **32 KB**

Total: **544 KB of TLS per thread**. On most platforms, the default stack size is 1–8 MB. If training is ever made multi-threaded or if the network architecture is enlarged, each thread would consume 544 KB of TLS, potentially causing stack-overflow crashes.

**Suggested Fix:**  
Heap-allocate these buffers using `std::vector` or `std::unique_ptr` with `aligned_alloc`, or document the single-threaded assumption clearly.

---

### 2.11 AVX2 Loops Assume `L1_SIZE` Is a Multiple of 8 (float) or 16 (int16)

**Severity:** 🟡 Medium  
**File:** NNUE.cpp  
**Lines:** 126–130

**Description:**  
`refreshAccumulator` uses `_mm256_load_ps`/`_mm256_store_ps` in loops incrementing by 8. `refreshAccumulatorQ` uses `_mm256_load_si256`/`_mm256_store_si256` incrementing by 16. If `L1_SIZE` is not a multiple of 16 (the stricter requirement), the loops read/write beyond the allocated array, causing undefined behaviour (likely a segfault). Currently `L1_SIZE = 512` is fine, but this is a latent bug if the architecture constant changes.

**Suggested Fix:**  
Add a `static_assert`:
```cpp
static_assert(L1_SIZE % 16 == 0,
    "L1_SIZE must be multiple of 16 for AVX2 int16 and float alignment");
```

---

### 2.12 AVX2 Aligned Load Requires 32-Byte Aligned Accumulators

**Severity:** 🟡 Medium  
**File:** NNUE.cpp  
**Lines:** 126–130

**Description:**  
`_mm256_load_ps` and `_mm256_store_ps` require 32-byte alignment. If the `Accumulator` struct or `QAccumulator` is stack-allocated without an explicit `alignas(32)` declaration, these SIMD loads/stores will fault with `SIGBUS` or generate an access violation on Windows. Similarly, `L1_biases` and each row of `L1_weights` must be 32-byte aligned.

**Suggested Fix:**  
Ensure all SIMD-accessed data carries the alignment declaration:
```cpp
struct alignas(32) Accumulator {
    alignas(32) float white[L1_SIZE];
    alignas(32) float black[L1_SIZE];
};
```
And verify weight arrays are allocated with `aligned_alloc(32, ...)`.

---

### 2.13 L3 Gradient and L2→L3 Backward Pass Use Scalar Loops

**Severity:** 🟡 Medium  
**File:** NNUETrainer.cpp  
**Lines:** 2360–2375

**Description:**  
The L3 weight gradient computation (lines 2360–2364) and the `dL/dL2Out` backward pass (lines 2370–2375) are fully scalar O(`L2_SIZE × L3_SIZE`) loops. With `L2_SIZE=128` and `L3_SIZE=64`, that is 8,192 multiply-adds per sample, performed twice per backward pass. The L2 gradient computation at lines 2386–2394 already uses AVX2 FMA — the L3 paths should follow the same pattern.

**Suggested Fix:**  
Vectorise with AVX2 broadcast + FMA, mirroring the L2 gradient pattern:
```cpp
// Example: L3 weight gradient
for (int j = 0; j < L3_SIZE; j++) {
    __m256 dj = _mm256_set1_ps(dLdL3Pre[j]);
    for (int i = 0; i < L2_SIZE; i += 8) {
        __m256 l2 = _mm256_load_ps(&l2Out[i]);
        __m256 g  = _mm256_load_ps(&L3_wgrad[j][i]);
        g = _mm256_fmadd_ps(dj, l2, g);
        _mm256_store_ps(&L3_wgrad[j][i], g);
    }
}
```

---

### 2.14 Training Phase Mapping Uses Discrete Labels Instead of Continuous Phase

**Severity:** 🟡 Medium  
**File:** NNUETrainer.cpp  
**Lines:** 2237–2242

**Description:**  
During training, game phase is mapped from discrete labels: `0 = opening → p=1.0`, `1 = middlegame → p=0.5`, `2 = endgame → p=0.0`. During inference (NNUE.cpp line 100–106), phase is a continuous value `board.phase / 24.0`. This mismatch means the network is trained on a 3-point quantised phase curve but evaluated on a smooth gradient. The network may learn phase-dependent weights that are miscalibrated at phase values never seen during training.

**Suggested Fix:**  
Store the actual material phase count in training data records and compute `p = phase / 24.0` during training, matching the inference path exactly.

---

### 2.15 `lastStatusPrefix` Thread Safety Is Correct but Fragile

**Severity:** 🟡 Medium  
**File:** SelfPlayGen.cpp  
**Lines:** 8360–8361, 8518, 8571

**Description:**  
`lastStatusPrefix` (char[256]) is written under `printMtx` lock by worker threads (line 8518) and read under the same lock by the countdown thread (line 8571). The synchronisation is currently correct. However, `lastEwmaGpsForDisplay` (double) is also written under lock but the pattern is fragile — a future maintainer could reasonably read it without the lock and introduce a data race.

**Suggested Fix:**  
No immediate fix required. Consider wrapping these display fields in a `struct StatusDisplay { std::mutex mtx; char prefix[256]; double ewmaGps; }` with named accessor methods to make the ownership semantics explicit and prevent accidental lock-free access.

---

### 2.16 `packWeights`/`unpackWeights` Use Scalar Loops for Millions of Parameters

**Severity:** 🟡 Medium  
**File:** NNUETrainer.cpp  
**Lines:** 1946–2005

**Description:**  
With `NUM_FEATURES ≈ 40960` and `L1_SIZE = 512`, the L1 weight matrix alone has ~20M float parameters. `packWeights` and `unpackWeights` copy every parameter one at a time in scalar loops. While only called once per epoch, for large networks this adds meaningful latency to epoch startup.

**Suggested Fix:**  
Where memory layout is contiguous (row-major arrays), use `memcpy` for each weight block:
```cpp
memcpy(flat.data() + offset, L1_weights.data(), numL1w * sizeof(float));
```

---

### 2.17 `forward()` Truncates Float to Int Instead of Rounding

**Severity:** 🟢 Low  
**File:** NNUE.cpp  
**Lines:** 394, 1071

**Description:**  
`static_cast<int>(output)` truncates toward zero. A score of 99.9 becomes 99, and −99.9 becomes −99, introducing a slight downward bias in absolute value. The same issue exists in `forwardQ()` at line 1071.

**Suggested Fix:**  
```cpp
return static_cast<int>(std::lround(output));
```

---

### 2.18 `evalWhitePOV` Perspective Assumption Undocumented

**Severity:** 🟢 Low  
**File:** NNUETrainer.cpp  
**Lines:** 1711

**Description:**  
`evalWhitePOV = static_cast<float>(eval)` where `eval` comes from `engine->getLiveEval()`. The variable name implies White POV, but whether `getLiveEval()` returns White POV or side-to-move POV is not documented. If the engine returns STM-relative eval, Black-to-move positions need negation to get White POV.

**Suggested Fix:**  
Add a comment clarifying the return convention of `getLiveEval()` and verify the conversion is correct. Add an assertion or test against a known position.

---

### 2.19 `releaseFloatWeights()` Invalidates Pointers Used by Float-Path Functions

**Severity:** 🟢 Low  
**File:** NNUE.cpp  
**Lines:** 424–426

**Description:**  
After `releaseFloatWeights()` sets `L1_weights.reset()`, any subsequent call to `addFeature()`, `removeFeature()`, `incrementalUpdate()`, or `refreshAccumulator()` will dereference a null pointer and crash. The quantised Q-variant path uses `L1_weights_q` (separate), but there is no runtime guard preventing accidental float-path use after release.

**Suggested Fix:**  
Add null-pointer guards in float-path functions:
```cpp
void addFeature(...) {
    assert(L1_weights && "addFeature called after releaseFloatWeights()");
    ...
}
```
Or document `releaseFloatWeights()` with a bold warning that all subsequent calls must use Q-variants only.

---

### 2.20 `refreshAccumulator` Scans All 64 Squares for Pieces

**Severity:** 🟢 Low  
**File:** NNUE.cpp  
**Lines:** 132–165

**Description:**  
`refreshAccumulator` iterates over all 64 mailbox squares to find active pieces. In endgame positions with few pieces, the majority of iterations are wasted. Since the `Board` maintains bitboards, piece iteration can use `popLsb` for O(pieces) traversal.

**Suggested Fix:**  
Iterate over `board.occupied()` using `popLsb`:
```cpp
Bitboard occ = board.occupied();
while (occ) {
    int sq = popLsb(occ);
    // add feature for piece at sq
}
```

---

### 2.21 Average Epoch Time Calculation Includes Initialisation Overhead for Epoch 0

**Severity:** 🟢 Low  
**File:** NNUETrainer.cpp  
**Lines:** 2531–2533

**Description:**  
When `epoch == 0`, `epochTime = elapsed` (total time including pre-epoch initialisation: pre-conversion, index building, etc.). For subsequent epochs, `epochTime = elapsed / (epoch+1)`. The first epoch's inflated time skews the ETA estimate for all subsequent epochs in the same run.

**Suggested Fix:**  
Record the start time of the first actual training epoch (after initialisation) and compute per-epoch duration from that point:
```cpp
auto epochStart = std::chrono::steady_clock::now();
// ... epoch ...
epochTime = duration_since(epochStart);
```

---

### 2.22 Phase Blend Uses Quadratic Bézier Weights — Behaviour Should Be Documented

**Severity:** 🟢 Low  
**File:** NNUE.cpp  
**Lines:** 379–383

**Description:**  
The phase blending uses Bernstein polynomial weights: `w_op = p²`, `w_eg = (1−p)²`, `w_mg = 2p(1−p)`. These sum to 1.0 (correct), but at the midpoint `p = 0.5`, the weights are 0.25/0.50/0.25 (opening/middlegame/endgame). This is a design choice, not a bug, but it is not documented. Future developers may assume linear interpolation and be surprised by the curve shape.

**Suggested Fix:**  
Add a comment explaining the blending scheme:
```cpp
// Bernstein quadratic basis: w_op=p^2, w_mg=2p(1-p), w_eg=(1-p)^2
// At p=0.5 (midgame): weights are 0.25/0.50/0.25
```

---

## Section 3: UCI, GUI & Tests

**Files:** UCI.cpp, GameLogic.cpp, VisualGame.cpp, main.cpp, Test.cpp, SmokeTest.cpp

---

### 3.1 Search Thread Races on Engine State After `getBestMove` Returns

**Severity:** 🟠 High  
**File:** UCI.cpp  
**Lines:** 320–356

**Description:**  
The search thread lambda captures `this` and calls `engine_.getLastDepth()`, `engine_.getLiveEval()`, `engine_.getPV()`, and `engine_.getPonderMove()` *after* `getBestMove()` returns. Between `getBestMove()` completing and these accessor calls executing, the main UCI thread could process a new `position` or `ucinewgame` command that modifies engine state. Although the code attempts to join the thread before modifying state, a rapid `stop` + new `position` + `go` sequence could create a window where the accessors race with a new search.

**Suggested Fix:**  
Capture all necessary engine results atomically immediately after `getBestMove()` returns, before releasing any implicit lock or continuing:
```cpp
auto [depth, eval, pv, ponder] = engine_.captureSearchResult(); // atomic snapshot
std::cout << "info depth " << depth << " score cp " << eval ...;
std::cout << "bestmove " << pv[0] << " ponder " << ponder;
```

---

### 3.2 `onInfoCallback` Captures `multiPV` by Value, Goes Stale Mid-Search

**Severity:** 🟠 High  
**File:** UCI.cpp  
**Lines:** 294–311

**Description:**  
The `onInfoCallback` lambda captures `multiPV` at the time `cmdGo` is called. If a GUI sends `setoption name MultiPV value N` while a search is running (which violates the UCI spec but is defensively handled elsewhere), the captured value becomes stale. The callback uses it to decide whether to print the `multipv` field in info strings, which could cause incorrect output (missing or spurious `multipv` tags).

**Suggested Fix:**  
Have the callback receive the current multiPV from the engine as a parameter, or lock out `setoption` changes during active search.

---

### 3.3 `setoption` Can Modify Engine During Active Search (Including `resizeTT`)

**Severity:** 🟡 Medium  
**File:** UCI.cpp  
**Lines:** 58–117

**Description:**  
The `setoption` handler calls `engine_.resizeTT()`, `engine_.setThreadCount()`, `engine_.setMultiPV()`, and `engine_.setContempt()` without first stopping and joining any active search thread. `resizeTT()` in particular reallocates the TT. If the search thread is concurrently probing or writing the TT, this is a data race that can cause heap corruption or a crash. Per the UCI spec, GUIs should not send `setoption` during search, but a defensive implementation must guard against it.

**Suggested Fix:**  
Apply the same stop+join guard used by `cmdPosition` and `cmdNewGame` before processing `setoption`.

---

### 3.4 Default of 5 Seconds When `go` Has No Parameters Violates UCI Spec

**Severity:** 🟡 Medium  
**File:** UCI.cpp  
**Lines:** 246–271

**Description:**  
When `go` is sent with no parameters (no `depth`, `movetime`, `wtime`/`btime`, or `infinite`), the engine defaults to a 5000 ms time limit. The UCI specification states that `go` with no parameters should search indefinitely. Some GUI configurations send bare `go` commands, which would cause the engine to stop prematurely.

**Suggested Fix:**  
Track whether any time-control parameter was provided; if none was, treat as `go infinite`:
```cpp
bool anyTimeParam = (movetime > 0 || wtime > 0 || btime > 0 || depth > 0 || infinite);
if (!anyTimeParam) infinite = true;
```

---

### 3.5 Time Allocation Uses OR Condition That May Use Unset Opponent Time

**Severity:** 🟡 Medium  
**File:** UCI.cpp  
**Lines:** 250–252

**Description:**  
Line 250: `if (wtime > 0 || btime > 0)` enters the time-allocation block for any non-zero time. If only `wtime` is provided but not `btime`, the engine playing as Black would compute `allocatedMs` based on `btime = 0`, resulting in only the increment being available. This effectively collapses Black's time allocation to zero plus increment, which can cause grossly incorrect time usage.

**Suggested Fix:**  
Only use the side-to-move's time value if it was explicitly provided. Track which parameters were actually set using an explicit `bool wtimeSet`, `bool btimeSet`.

---

### 3.6 Engine Thread References `boardCopy` Without Lifetime Guarantee

**Severity:** 🟡 Medium  
**File:** VisualGame.cpp  
**Lines:** 1810–1817

**Description:**  
The engine thread lambda captures `boardCopy` by value, but if `getBestMove` internally stores a pointer or reference to the `Board` parameter (e.g., for `isRepetition` checks or ponder), that reference would be to the lambda's local copy. While this appears safe today, a future change to `getBestMove`'s signature or internal storage that retains a board reference after returning could cause use-after-free.

**Suggested Fix:**  
Verify explicitly in a comment (and/or with a static analysis annotation) that `getBestMove` does not retain any reference to the `Board` parameter after returning. Consider passing by value rather than by reference to `getBestMove`.

---

### 3.7 Move Number Increment Timing Should Be Documented

**Severity:** 🟡 Medium  
**File:** VisualGame.cpp  
**Lines:** 1697–1698

**Description:**  
The move number is incremented when `board.turn == Color::Black` *before* calling `board.applyMove()`. This is correct for standard chess notation (increment after Black completes a move), but the logic is non-obvious: at the point of the check, `board.turn` is the side *currently making* the move (Black), not the side that has just finished. A reader who sees `if (board.turn == Black) moveNumber++` before `applyMove` might incorrectly interpret it.

**Suggested Fix:**  
Add a clarifying comment:
```cpp
// Increment after Black's move (before applyMove, board.turn is the side about to move)
if (board.turn == Color::Black) moveNumber++;
```

---

### 3.8 `isInsufficientMaterial` Uses Mailbox Scan Instead of Bitboards

**Severity:** 🟡 Medium  
**File:** GameLogic.cpp  
**Lines:** 547–598

**Description:**  
`isInsufficientMaterial` iterates all 64 squares with nested loops to count pieces by type. Functionally correct for the single-bishop cases it checks, but unnecessarily slow. Additionally, K+N+N vs K is not recognised as a draw (it is technically a mate risk but generally treated as drawn since it requires opponent cooperation).

**Suggested Fix:**  
Use bitboard popcount for piece counting:
```cpp
int whiteBishops = popcount(board.pieces(White, Bishop));
int whiteKnights = popcount(board.pieces(White, Knight));
// etc.
```
Consider also adding K+N+N vs K to the draw table.

---

### 3.9 `updateStatus` Reimplements `GameLogic::classify`, Risks Divergence

**Severity:** 🟡 Medium  
**File:** VisualGame.cpp  
**Lines:** 1919–1995

**Description:**  
`VisualGame::updateStatus()` manually reimplements threefold-repetition detection, 50-move rule, checkmate, and stalemate checks. `GameLogic::classify()` already encapsulates all of these correctly. Having two independent implementations means any future change to draw/win detection must be applied in both places. Currently, `updateStatus()` also *omits* the insufficient-material draw check entirely (see finding 3.10).

**Suggested Fix:**  
Replace the body of `updateStatus()` with a call to `GameLogic::classify(board, positionHistory)` and update the UI based on the returned `GameResult` enum.

---

### 3.10 GUI Does Not Detect Draws by Insufficient Material

**Severity:** 🟡 Medium  
**File:** VisualGame.cpp  
**Lines:** 1939–1950

**Description:**  
`updateStatus()` checks for threefold repetition, 50-move rule, checkmate, and stalemate, but never checks for insufficient material (K vs K, K+B vs K, K+N vs K, K+B vs K+B same-colour). Games ending in these configurations will continue indefinitely in the GUI until the 50-move rule triggers.

**Suggested Fix:**  
Add to `updateStatus()`:
```cpp
if (GameLogic::isInsufficientMaterial(board)) {
    gameResult = GameResult::DrawInsufficient;
    return;
}
```
Or preferably, address finding 3.9 first (use `GameLogic::classify()`) which will include this check automatically.

---

### 3.11 `updateETA` Thread Safety Is Correct but Superficially Fragile

**Severity:** 🟡 Medium  
**File:** VisualGame.cpp  
**Lines:** 903–912

**Description:**  
`updateETA()` is called from background threads (training, ELO estimation) and writes to `nnueETAEndMs_`. The atomic store provides the necessary visibility to the main rendering thread. The calculation uses only lambda-captured start time and callback parameters, which are thread-local. No fix needed — the current implementation is correct.

**Suggested Fix:**  
No change required. Consider adding a comment: `// nnueETAEndMs_ is atomic; reads from render thread are safe without additional locking.`

---

### 3.12 `fastMode` Animation Skip Inconsistent Between `botVsBot` and `botVsNNUE`

**Severity:** 🟡 Medium  
**File:** VisualGame.cpp  
**Lines:** 1656–1657, 1875

**Description:**  
In `executeMove()`, line 1656: `if (fastMode && botVsBot)` skips animations. `checkEngineResult()` at line 1875 correctly checks `(botVsBot || botVsNNUE_)`. The inconsistency means `botVsNNUE_` games in fast mode still play animations while the engine result check skips the delay.

**Suggested Fix:**  
```cpp
if (fastMode && (botVsBot || botVsNNUE_)) {
    // skip animation
}
```

---

### 3.13 Redundant Info Line After Search Duplicates Per-Depth Callback Output

**Severity:** 🟢 Low  
**File:** UCI.cpp  
**Lines:** 339–344

**Description:**  
After `getBestMove()` returns, the search thread prints another info line with the final depth/score/PV. This duplicates the info line already sent by `onInfoCallback` at the last completed depth. Some GUIs display both lines, showing what appears to be two searches completing at the same depth.

**Suggested Fix:**  
Remove the post-search info line, since the callback already sends info at each depth. Alternatively, only send it if the final result differs from the last callback depth.

---

### 3.14 Post-Search Info Line Hardcodes `multipv 1` Regardless of Actual Index

**Severity:** 🟢 Low  
**File:** UCI.cpp  
**Lines:** 340

**Description:**  
When `multiPV > 1`, the final info line always prints `multipv 1`. This is potentially incorrect and misleading, especially when the best move did not come from PV line 1 in the engine's internal ordering.

**Suggested Fix:**  
Either report all PV lines in the final output, or omit the `multipv` tag from the final summary line.

---

### 3.15 UCI Whitespace Trimming Uses O(n²) `erase` in Loop

**Severity:** 🟢 Low  
**File:** UCI.cpp  
**Lines:** 36–39

**Description:**  
Leading whitespace is trimmed by calling `line.erase(line.begin())` in a loop. Each `erase` at the front of a `std::string` is O(n), making the overall trim O(n²) in the number of leading spaces. While UCI lines are short in practice, this is easily avoided.

**Suggested Fix:**  
```cpp
auto it = std::find_if(line.begin(), line.end(),
    [](unsigned char c) { return !std::isspace(c); });
line.erase(line.begin(), it);
```

---

### 3.16 Human Input Blocked While Engine Is Thinking

**Severity:** 🟢 Low  
**File:** VisualGame.cpp  
**Lines:** 786

**Description:**  
Line 786 blocks all mouse input when `engineThinking` is true, preventing the player from pre-selecting pieces or planning moves during the engine's think time. This is a UX limitation, not a bug.

**Suggested Fix:**  
Consider allowing piece *selection* (but not move *execution*) while the engine is thinking.

---

### 3.17 Promotion Dialog Sprite Scaling Uses `ts.x` Instead of `max(ts.x, ts.y)`

**Severity:** 🟢 Low  
**File:** VisualGame.cpp  
**Lines:** 2299

**Description:**  
In `drawPromotionDialog()`, the sprite scale is `float(SQ) / float(ts.x)`. All other sprite scaling in the codebase uses `float(SQ) / float(std::max(ts.x, ts.y))` to ensure non-square textures are correctly letterboxed. Non-square promotion piece textures would render stretched/compressed.

**Suggested Fix:**  
```cpp
float scale = float(SQ) / float(std::max(ts.x, ts.y));
```

---

### 3.18 `releaseFloatWeights` Called Unconditionally Even When `loadWeights` Fails

**Severity:** 🟢 Low  
**File:** main.cpp  
**Lines:** 2974

**Description:**  
Line 2974 calls `nnue->releaseFloatWeights()` unconditionally after `loadWeights()`. If `loadWeights()` failed (the warning at lines 2970–2973 prints a message but continues), `releaseFloatWeights()` is called on an unloaded/uninitialised network, which may be a no-op or may access uninitialised memory depending on the NNUE implementation.

**Suggested Fix:**  
```cpp
if (weightsLoaded) {
    nnue->releaseFloatWeights();
}
```

---

### 3.19 Windows Pipe I/O: `*stdout = *fpOut` Is Implementation-Defined

**Severity:** 🟢 Low  
**File:** main.cpp  
**Lines:** 2756

**Description:**  
Assigning `FILE` structs via `*stdout = *fpOut` is not portable — it relies on MSVC's `FILE` being a simple copyable struct. This is implementation-defined behaviour under the C standard and could break with a different CRT version or compiler.

**Suggested Fix:**  
Use `_dup2` for more robust file-handle redirection:
```cpp
_dup2(_fileno(fpOut), _fileno(stdout));
```
Or use `freopen_s`.

---

### 3.20 Expected-Move Mismatch Counted as PASS Instead of WARN/FAIL

**Severity:** 🟢 Low  
**File:** SmokeTest.cpp  
**Lines:** 3464–3471

**Description:**  
When the engine finds a legal move but not the expected one (line 3466), it increments `passed++` with only a `WARN` log message. This means the test suite exit code reports success even when the engine fails to find expected moves like mate-in-1. Test assertions are effectively advisory only.

**Suggested Fix:**  
Add a separate `warned` counter. Consider failing on critical position mismatches (mate-in-1) or provide a `--strict` mode that counts warnings as failures.

---

### 3.21 Perft Should Verify `applyMove` and `makeMove` Produce Identical Board State

**Severity:** 🟢 Low  
**File:** Test.cpp  
**Lines:** 3052–3068

**Description:**  
Perft correctly uses `board.makeMove()`/`board.unmakeMove()`. The GUI uses `board.applyMove()` instead. If `makeMove` and `applyMove` ever produce different board states (e.g., different hash, different material counts), perft would pass while GUI behaviour is wrong. There is no test verifying the two paths are equivalent.

**Suggested Fix:**  
Add a test that for each legal move from a set of representative positions, applies the move with both `makeMove` and `applyMove` and asserts that all fields of the resulting `Board` are equal.

---

### 3.22 Threefold Repetition in `GameLogic` Always Scans Entire History

**Severity:** 🟢 Low  
**File:** GameLogic.cpp  
**Lines:** 534–545

**Description:**  
`isThreefoldRepetition()` linearly scans the entire `positionHistory` vector on every call. For long games (hundreds of moves), the cost grows proportionally. The scan could be short-circuited at the last irreversible move (the `halfMoveClock` resets to 0 on pawn moves or captures; repetitions cannot cross an irreversible move).

**Suggested Fix:**  
Only scan back `halfMoveClock` positions:
```cpp
int start = std::max(0, (int)positionHistory.size() - board.halfMoveClock - 1);
for (int i = start; i < (int)positionHistory.size(); i++) { ... }
```

---

### 3.23 `VisualGame.cpp` Is Extremely Large and Mixes Unrelated Concerns

**Severity:** ℹ️ Info  
**File:** VisualGame.cpp  
**Lines:** 685–2714

**Description:**  
`VisualGame.cpp` (~2,000 lines) combines rendering, input handling, engine thread management, NNUE training orchestration, ELO estimation, animation, and HUD rendering in a single file. This makes navigation, testing, and maintenance difficult. Adding a new rendering feature requires reading the engine management code, and vice versa.

**Suggested Fix:**  
Consider splitting into: `VisualGameRender.cpp`, `VisualGameInput.cpp`, `VisualGameEngine.cpp`, `VisualGameTraining.cpp`, each with a corresponding header and clearly delimited responsibilities.

---

### 3.24 `Press Enter to Exit` Blocks CI/CD Pipelines

**Severity:** ℹ️ Info  
**File:** Test.cpp  
**Lines:** 3223–3224

**Description:**  
Both perft and smoke test suites end with `"Press Enter to exit..."` followed by `std::cin.get()`. This blocks any CI/CD pipeline that does not simulate terminal input, causing automated test jobs to hang.

**Suggested Fix:**  
Make the prompt conditional:
```cpp
if (isatty(fileno(stdin))) {
    std::cout << "Press Enter to exit..."; std::cin.get();
}
```
Or add a `--no-wait` command-line flag.

---

### 3.25 UCI Auto-Detection Described in Comment but Not Implemented

**Severity:** ℹ️ Info  
**File:** main.cpp  
**Lines:** 2986–3004

**Description:**  
The comment at line 2986 describes first-stdin-line auto-detection as a way for GUIs to enter UCI mode without the `--uci` flag. The actual code only checks `argv` for `--uci`. The auto-detection is not implemented, making the comment misleading.

**Suggested Fix:**  
Either implement stdin auto-detection:
```cpp
std::string firstLine;
if (std::getline(std::cin, firstLine) && firstLine == "uci") {
    enterUCIMode(firstLine);
}
```
Or remove the misleading comment.

---

## Section 4: Build System & Scripts

**Files:** train_nnue.py, pgn_to_training.py, training_format.py, pipeline.ps1, smart_push.ps1, elo_calibration.ps1, pgo_build.sh, train.bat, build_tests.bat, CMakeLists.txt

---

### 4.1 `--extra-data` Argument Has No Validation — Crashes with Unhelpful Error on Misuse

**Severity:** 🟠 High  
**File:** train_nnue.py  
**Lines:** 2266–2292

**Description:**  
`--extra-data` expects alternating `path ratio` pairs (e.g., `--extra-data selfplay.bin 0.2`), but `argparse nargs='*'` accepts any number of values. If a user passes an odd number of items (e.g., only a path with no ratio), the last path is silently paired with a default ratio of 0.5. If a ratio value is a typo (e.g., `abc` instead of `0.2`), the code crashes deep in a `float()` conversion with an unhelpful `ValueError` and no indication of which argument is wrong.

**Suggested Fix:**  
```python
extra = args.extra_data
if len(extra) % 2 != 0:
    parser.error("--extra-data requires alternating path and ratio pairs")
for i in range(1, len(extra), 2):
    try:
        ratio = float(extra[i])
    except ValueError:
        parser.error(f"--extra-data: invalid ratio '{extra[i]}' (expected a float)")
    if not 0.0 <= ratio <= 1.0:
        parser.error(f"--extra-data: ratio {ratio} out of range [0, 1]")
```
Consider using `nargs=2, action='append'` instead.

---

### 4.2 Best-Loss Checkpoint and Periodic Checkpoint Can Save Inconsistent State

**Severity:** 🟠 High  
**File:** train_nnue.py  
**Lines:** 4847–4866

**Description:**  
The best-loss checkpoint (line 4847) saves state when `val_loss` improves and resets `epochs_no_improve = 0`. The periodic checkpoint (line 4866) saves on `epochs_done % save_every == 0`. On a non-improving epoch, `epochs_no_improve` is incremented at line 4861 *after* the best-loss block but *before* the periodic save. If training is resumed from the best-loss checkpoint (not the periodic one), `epochs_no_improve` is reset to 0, potentially allowing training to continue well beyond what early-stopping intended.

**Suggested Fix:**  
Consolidate checkpoint saving into a single canonical location after all counters are updated:
```python
# After all counters updated:
if val_loss < best_val_loss:
    save_checkpoint(path=best_loss_path, state=get_state())
if epochs_done % save_every == 0:
    save_checkpoint(path=periodic_path, state=get_state())
```

---

### 4.3 Training Data Format Has Undocumented Endianness Dependency

**Severity:** 🟠 High  
**File:** train_nnue.py / training_format.py  
**Lines:** 1600–1700

**Description:**  
The Python training data reader uses `struct.unpack` with `'<'` (little-endian) format codes. The C++ `SelfPlayGen` writes using native byte order with no explicit endian conversion. On x86 (little-endian) this works. On a big-endian platform (e.g., some ARM configurations, PowerPC), the C++ writer would produce big-endian data that Python reads as little-endian, producing completely garbled positions with no diagnostic error.

**Suggested Fix:**  
Either document the little-endian requirement explicitly in both the C++ and Python code, or add a magic number/endianness marker at the start of the binary file and validate it on read:
```cpp
// C++ writer: explicitly write as little-endian
uint32_t magic = htole32(0x4E4E5545); // "NNUE"
fwrite(&magic, sizeof(magic), 1, fp);
```

---

### 4.4 Self-Play Output File Not Validated for Corruption

**Severity:** 🟠 High  
**File:** pipeline.ps1  
**Lines:** 7246–7252

**Description:**  
The pipeline detects self-play failure only by exit code and file existence. A partially written or corrupted binary file (e.g., from a crash mid-write) passes this check. A truncated file could contain only a fraction of the expected positions, silently degrading training quality without any error. There is no file-size sanity check, position-count validation, or checksum verification.

**Suggested Fix:**  
Add a minimum file size check:
```powershell
$minExpectedBytes = $cfg.GamesPerGen * $cfg.AvgPositionsPerGame * $recordSize
if ((Get-Item $selfplayFile).Length -lt $minExpectedBytes) {
    Write-Error "Self-play output file is too small — possible corruption"
    exit 1
}
```
Long-term, add a header to the self-play binary format with position count and a simple checksum.

---

### 4.5 SWA Model Never Validated; Exported Model May Differ from Best Validated Model

**Severity:** 🟡 Medium  
**File:** train_nnue.py  
**Lines:** 3700–3750

**Description:**  
The SWA (Stochastic Weight Averaging) model is updated each epoch, but validation loss is always computed on the *non-averaged* model. The best checkpoint saved via `save_weights_cpp` uses non-averaged weights and corresponds to the validated best `val_loss`. However, at end of training, the final exported model uses SWA-averaged weights — which were never validated. The metric used to compare models (`val_loss`) corresponds to different weights than what is deployed.

**Suggested Fix:**  
After SWA concludes, run one final validation pass on the SWA-averaged model and log its loss. Only export the SWA model if its validation loss is better than the best non-SWA checkpoint. Consider saving both models with clear labels.

---

### 4.6 Weight Quantisation Has No Precision-Loss Reporting

**Severity:** 🟡 Medium  
**File:** train_nnue.py  
**Lines:** 2100–2150

**Description:**  
The `save_weights_cpp` function quantises `float32` weights to `int16` with a fixed global scaling factor. Small but important weights near zero may be quantised to 0, silently pruning connections. With a single scale factor for all layers, layers with very different weight magnitude ranges (e.g., L1 vs L3) may have different quantisation errors with no visibility into how much precision is lost.

**Suggested Fix:**  
Log quantisation statistics after export:
```python
for name, weights, quant_weights in weight_pairs:
    max_err = np.abs(weights - quant_weights / scale).max()
    zero_frac = (quant_weights == 0).mean()
    print(f"{name}: max_quant_error={max_err:.4f}, zeroed_weights={zero_frac:.2%}")
```
Consider per-layer scaling factors for better precision.

---

### 4.7 VS Discovery in `train.bat` Uses Fragile Hardcoded Paths

**Severity:** 🟡 Medium  
**File:** pipeline.ps1 / train.bat  
**Lines:** 7730–7742 (pipeline.ps1)

**Description:**  
`train.bat` searches for Visual Studio in hardcoded paths like `C:\Program Files\Microsoft Visual Studio\{year}\{edition}\VC\Auxiliary\Build\vcvarsall.bat`. This won't find VS in non-standard install locations. `build_tests.bat` already uses `vswhere.exe` for reliable VS discovery — `train.bat` should be unified with this approach.

**Suggested Fix:**  
```batch
for /f "tokens=*" %%i in ('"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath') do set VS_PATH=%%i
call "%VS_PATH%\VC\Auxiliary\Build\vcvarsall.bat" x64
```

---

### 4.8 `smart_push.ps1` Force-Push Script Has Insufficient Validation of File Paths

**Severity:** 🟡 Medium  
**File:** smart_push.ps1  
**Lines:** 7436–7511

**Description:**  
`smart_push.ps1` uses `--force-with-lease` (good), but automatically modifies `.gitignore` and commits before pushing. The script uses `Get-ChildItem` to discover files, then constructs gitignore patterns from their paths. The regex-escaping logic may not handle all edge cases (e.g., paths with backticks, dollar signs, or single quotes, which are PowerShell metacharacters). A file with a crafted name could cause the gitignore pattern to match unintended files.

**Suggested Fix:**  
Validate discovered file paths against a whitelist of expected characters before passing to `git` commands. Consider using `git lfs` for large binary file management instead of force-push exclusion.

---

### 4.9 Elo Calculation Does Not Apply `MaxSyntheticEloDiff` Cap in All Branches

**Severity:** 🟡 Medium  
**File:** elo_calibration.ps1  
**Lines:** 6393–6410

**Description:**  
The fallback Elo calculation at line 6407: `-400 * [Math]::Log10(1/$score - 1)`. When `score` is very close to 0 or 1 (but not exactly), the result can be extremely large (e.g., score=0.001 → Elo=1200). The `MaxSyntheticEloDiff` cap is applied in the `score=0` and `score=1` special-case branches, but the general formula branch at line 6407 does not apply the cap.

**Suggested Fix:**  
Apply the cap after the formula:
```powershell
$elo = [Math]::Max(-$MaxSyntheticEloDiff,
        [Math]::Min($MaxSyntheticEloDiff,
            -400 * [Math]::Log10(1/$score - 1)))
```

---

### 4.10 Gradient Accumulation Step Counter Not Saved in Checkpoint

**Severity:** 🟡 Medium  
**File:** train_nnue.py  
**Lines:** 4700–4750

**Description:**  
The training loop uses gradient accumulation (`grad_accum` steps). `global_step` is saved in checkpoints, but the position *within* a gradient accumulation cycle is not. If training is interrupted and resumed mid-accumulation (e.g., `global_step % grad_accum == 2` out of 4), the optimizer step timing will be off for the first resumed batch, effectively changing the learning rate for that batch.

**Suggested Fix:**  
Ensure `global_step` only increments on optimizer steps (not on accumulation sub-steps). This makes the checkpoint state unambiguous:
```python
# increment global_step only when optimizer.step() is called
if (batch_idx + 1) % grad_accum == 0:
    optimizer.step()
    global_step += 1
```

---

### 4.11 Training Data Loaded Entirely Into RAM — No Memory-Mapped Option

**Severity:** 🟡 Medium  
**File:** train_nnue.py  
**Lines:** 4200–4300

**Description:**  
All training data (base + all extra-data files) is loaded entirely into memory. For large datasets (millions of positions), this can consume many GB of RAM. With multiple extra-data sources, memory usage can exceed system RAM, causing OOM errors or heavy swapping with no warning.

**Suggested Fix:**  
Add a memory-mapped option for large datasets:
```python
if use_mmap:
    data = np.memmap(path, dtype=np.uint8, mode='r')
else:
    data = np.frombuffer(open(path, 'rb').read(), dtype=np.uint8)
```
Document the memory requirements in the README based on dataset size.

---

### 4.12 PGN Parser May Not Handle All Edge Cases in Move Notation

**Severity:** 🟡 Medium  
**File:** pgn_to_training.py  
**Lines:** 200–350

**Description:**  
The custom PGN parser is likely missing handling for: promotion with check (`e8=Q+`), null moves (`--`), long algebraic notation used by some exporters, and nested variation annotations. When an unparseable move is encountered, the error handling behaviour (skip game vs. skip move vs. crash) is unclear from the code.

**Suggested Fix:**  
Add explicit try/except around move parsing with game-level skip and counter:
```python
try:
    move = parse_move(token, board)
except ParseError as e:
    skipped_games += 1
    break  # skip rest of this game
```
Long-term, consider using the `python-chess` library's PGN parser, which handles all edge cases correctly.

---

### 4.13 AVX2 Enabled by Default with No Runtime Detection

**Severity:** 🟡 Medium  
**File:** CMakeLists.txt  
**Lines:** 7809–7816

**Description:**  
`CMakeLists.txt` enables AVX2 by default (`option(USE_AVX2 ON)`). If a user builds and runs on a CPU without AVX2 support (pre-2013 Intel, some AMD, budget cloud VMs), the binary will crash with `SIGILL` (illegal instruction) or `STATUS_ILLEGAL_INSTRUCTION` on Windows with no helpful diagnostic message.

**Suggested Fix:**  
Add a startup CPUID check in `main()`:
```cpp
if (!__builtin_cpu_supports("avx2")) {
    std::cerr << "Error: This build requires AVX2 support. "
                 "Rebuild with -DUSE_AVX2=OFF for a compatible binary.\n";
    return 1;
}
```
Also provide a non-AVX2 build configuration in the README.

---

### 4.14 `chess_core` Library Includes Training Code in the Engine Binary

**Severity:** 🟢 Low  
**File:** CMakeLists.txt  
**Lines:** 7822–7832

**Description:**  
The `chess_core` static library includes `NNUETrainer.cpp` and `SelfPlayGen.cpp`. The UCI engine executable (`chess_engine`) links against `chess_core` and thus includes several hundred KB of training code that is never used at runtime, unnecessarily increasing binary size.

**Suggested Fix:**  
Split the library:
- `chess_core`: Board.cpp, MoveGen.cpp, Bitboard.cpp, Engine.cpp, NNUE.cpp, UCI.cpp, GameLogic.cpp
- `chess_training`: NNUETrainer.cpp, SelfPlayGen.cpp, TrainingRunner.cpp

Link `chess_engine` against only `chess_core`; link the training executable against both.

---

### 4.15 Warmup Steps Calculated from `MaxPositions` Instead of Actual Dataset Size

**Severity:** 🟡 Medium  
**File:** pipeline.ps1  
**Lines:** 7140–7147

**Description:**  
Warmup steps are auto-calculated as `3 × ⌈MaxPositions / (BatchSize × GradAccum)⌉`. However, `MaxPositions` is the maximum number of positions to *load*, not the actual available count. If the training data has fewer positions than `MaxPositions`, the actual steps per epoch are fewer than calculated, and warmup extends beyond the intended 3 epochs.

**Suggested Fix:**  
Have `train_nnue.py` print the actual loaded dataset size and compute warmup from that:
```powershell
$actualPositions = (python -c "import training_format; print(training_format.count_positions('$dataFile'))")
$warmupSteps = 3 * [Math]::Ceiling($actualPositions / ($cfg.BatchSize * $cfg.GradAccum))
```

---

### 4.16 Early-Stop Time Normalisation Assumes Linear Per-Epoch Timing

**Severity:** 🟢 Low  
**File:** pipeline.ps1  
**Lines:** 7380–7400

**Description:**  
When training early-stops at epoch *k* out of *N*, the pipeline normalises the elapsed time by `N/k` to estimate what full training would have taken. This assumes each epoch takes equal time. In practice, later epochs may be slower (SWA overhead, validation computation, larger learning rate causing more loss variation). ETA estimates during the pipeline run will be inaccurate.

**Suggested Fix:**  
Use per-epoch timing data from `training_log.csv`, which the trainer already writes. Average the last few epoch durations and project forward.

---

### 4.17 PGO Build Script May Delete Profile Data Before It Is Merged (Clang Path)

**Severity:** 🟡 Medium  
**File:** pgo_build.sh  
**Lines:** 7531–7590

**Description:**  
For the Clang path, the script runs `llvm-profdata merge` at line 7560, then `rm -rf $BUILD_DIR` before recreating it. The `llvm-profdata merge` output file is written to the output path *outside* the build directory, so the merge result is safe. However, if `default.profraw` was written to the *current working directory* (which defaults to the build directory in some configurations), `rm -rf $BUILD_DIR` could delete the raw profile data before merge — if the script fails between creating the raw profile and running merge.

**Suggested Fix:**  
Move `llvm-profdata` output to `/tmp/` before deleting the build directory:
```bash
llvm-profdata merge -output=/tmp/pgo_profile.profdata *.profraw
rm -rf "$BUILD_DIR"
```

---

### 4.18 Dashboard Web Server Binds to All Interfaces with No Authentication

**Severity:** 🟢 Low  
**File:** train.bat  
**Lines:** 7767

**Description:**  
The training dashboard is launched on port 8080, binding to all network interfaces. On a shared system or cloud VM, this exposes training status (and potentially write endpoints if any exist) to any host that can reach port 8080.

**Suggested Fix:**  
Bind to localhost only by passing `--host 127.0.0.1` to the dashboard server, or document the security implication prominently.

---

### 4.19 `TrainingRunner.exe` Receives `PRESET` Quoted as Single Argument

**Severity:** 🟢 Low  
**File:** train.bat  
**Lines:** 7777

**Description:**  
`TrainingRunner.exe "%PRESET%"` wraps the entire `PRESET` variable in quotes. If `PRESET` is a multi-word string (e.g., `--full --extra-data file.bin 0.2`), it is passed as a single quoted argument rather than separate arguments, causing the training runner to fail to parse the command line.

**Suggested Fix:**  
Remove the quotes to allow normal argument splitting:
```batch
TrainingRunner.exe %PRESET%
```

---

### 4.20 Test Coverage Gap: No Unit Tests for Training Pipeline or NNUE I/O

**Severity:** 🟡 Medium  
**File:** build_tests.bat / CMakeLists.txt  
**Lines:** 7647–7689 / 7848–7854

**Description:**  
The test infrastructure covers perft (move generation correctness) and smoke tests (search sanity), but there are no tests for:
- NNUE weight export/import roundtrip (save then load and verify identical outputs)
- Training data format compatibility between C++ writer and Python reader
- Move generation edge cases (en passant capture, castling through check, promotions with check)
- Evaluation consistency (NNUE incremental vs. full refresh should produce identical results)
- Python training scripts (no `pytest` suite)

**Suggested Fix:**  
Add a C++ test that writes a training data file with known positions and verifies Python can read them identically. Add a C++ test that applies `makeMove`/`incrementalUpdate` and `refreshAccumulator` to the same position and asserts the same eval. Add a minimal `pytest` suite for `train_nnue.py` covering data loading, loss shape, and weight quantisation error bounds.

---

### 4.21 Training Arguments Built as Flat Array with Implicit Positional Coupling

**Severity:** 🟡 Medium  
**File:** pipeline.ps1  
**Lines:** 7263–7286

**Description:**  
Training arguments are built as a flat PowerShell array passed to `python train_nnue.py`. The `--extra-data` argument consumes the following values via `nargs='*'`, meaning the correctness of argument parsing depends on array element ordering. Adding a new argument between `--extra-data` and its path/ratio values would silently break the parsing.

**Suggested Fix:**  
Build `--extra-data` as a single pre-formatted string, or add explicit validation in `train_nnue.py` that `--extra-data` always receives pairs (see finding 4.1). Use named `--key=value` syntax wherever possible to eliminate positional coupling.

---

### 4.22 Argument Quoting in `Run-WithETA` May Break on Paths with Metacharacters

**Severity:** 🟢 Low  
**File:** pipeline.ps1  
**Lines:** 6863–6868

**Description:**  
The argument escaping logic checks for special characters and adds quotes, but may not handle paths containing PowerShell metacharacters (`$`, `` ` ``, `'`). A binary or data file with a crafted name could cause incorrect argument passing to child processes.

**Suggested Fix:**  
Use `Start-Process` with `-ArgumentList` as an array rather than a manually constructed string:
```powershell
Start-Process -FilePath $exe -ArgumentList $argArray -Wait -NoNewWindow
```
Or use `[System.Diagnostics.ProcessStartInfo]` with `ArgumentList` (available in .NET 5+).

---

### 4.23 Elo File Written with UTF-8 BOM on Windows PowerShell 5.x

**Severity:** 🟢 Low  
**File:** elo_calibration.ps1  
**Lines:** 6556–6563

**Description:**  
`Out-File -Encoding utf8` on Windows PowerShell 5.1 writes UTF-8 with a Byte Order Mark (BOM: `EF BB BF`). If the Elo file is read by a tool expecting plain UTF-8 or ASCII (e.g., a Linux script, a Python `open()` without `encoding='utf-8-sig'`, or some text parsers), the leading BOM bytes cause parsing failures or produce a leading `ï»¿` artefact.

**Suggested Fix:**  
Use BOM-free output:
```powershell
[System.IO.File]::WriteAllText($EloFile, $finalElo.ToString(), 
    [System.Text.UTF8Encoding]::new($false))
```
Or use PowerShell 7+ where `Out-File -Encoding utf8` does not emit BOM by default.

---

## Cross-Cutting Concerns

### Thread Safety

The codebase has an inconsistent approach to thread safety. UCI.cpp relies on the UCI protocol's implicit sequencing (no setoption during search) but does not enforce this defensively — `resizeTT` during an active search is a potential crash (finding 3.3). VisualGame.cpp uses a `std::mutex` + `std::atomic<bool>` pattern for the engine thread result, which is correct but somewhat redundant (findings 3.6, 3.11). SelfPlayGen.cpp is generally well-synchronised. The most serious thread-safety issue is the window between `getBestMove()` returning and the search thread reading engine state (finding 3.1).

**Recommendation:** Adopt a consistent pattern for communicating search results from the engine thread to the caller: either a dedicated result struct returned by `getBestMove()`, or an explicit `std::promise`/`std::future` pair.

### Error Handling

Most failure modes are handled with silent fallbacks rather than explicit errors. `loadWeights` failure continues with an uninitialised network (finding 3.18). `evaluateMovesNNUE` has an inverted score perspective with no diagnostic (finding 2.6). Self-play output corruption is not detected (finding 4.4). The training pipeline's `--extra-data` parsing silently defaults malformed input (finding 4.1).

**Recommendation:** Adopt a policy: all functions that can fail in ways that affect search quality or training data integrity must return an explicit error value (or throw, or assert) rather than silently continuing with wrong state.

### Portability

The codebase is primarily Windows-targeted but appears intended to also build on Linux:
- `_aligned_malloc`/`_aligned_free` (finding 2.5) will not compile on GCC/Clang
- `*stdout = *fpOut` FILE struct copy (finding 3.19) is MSVC implementation-dependent
- Little-endian training data format assumption (finding 4.3) would silently corrupt on big-endian
- AVX2 crash-on-unsupported-CPU with no diagnostic (finding 4.13)

**Recommendation:** Add a CI build on Linux (GCC 12+) to catch portability regressions early. The `_aligned_malloc` fix is the most urgent as it is a compile error.

### Test Coverage

The test suite (perft + smoke tests) validates move generation correctness and basic search sanity, but does not cover:
- NNUE evaluation correctness (incremental vs full refresh agreement)
- Training data format roundtrip
- Search correctness (mate-in-N, draw detection, repetition handling)
- UCI protocol compliance

**Recommendation:** Add at minimum: (1) an NNUE incremental/refresh consistency test, (2) a training data roundtrip test between C++ and Python, and (3) a search correctness suite against known positions (EPD format).

### Documentation

Several non-obvious design decisions are undocumented: the `applyMove` vs `makeMove` contract (finding 1.8), the `fusedCopyAndUpdateQ` pre/post-move board convention (finding 2.9), the phase blending curve (finding 2.22), and the `releaseFloatWeights` usage contract (finding 2.19). The UCI auto-detection comment describes functionality that doesn't exist (finding 3.25).

**Recommendation:** Add a `ARCHITECTURE.md` document describing: (1) the incremental NNUE update contract and which call sites use which path, (2) the training data binary format with a diagram, (3) the TT replacement and aging strategy, and (4) the thread model for UCI and VisualGame.

---

## Priority Action Plan

The following actions are ordered by impact-to-effort ratio — highest impact, lowest effort first.

1. **[Critical, ~1h] Fix TT mate score ply normalisation (Finding 1.1):** Add store/retrieve ply adjustment for mate scores in both `search()` and `qsearch()`. This is a well-understood fix with a precise formula. Impacts every game that involves mating sequences.

2. **[High, ~2h] Fix `_aligned_malloc` portability (Finding 2.5):** Replace with `std::aligned_alloc`/`std::free` wrapped in a platform macro. Required for Linux/CI builds to succeed.

3. **[High, ~1h] Fix TT best-move preservation on fail-high (Finding 1.2):** Remove the move-preservation logic at lines 1527–1531. One-line fix with clear move-ordering benefit.

4. **[High, ~2h] Fix `evaluateMovesNNUE` score perspective (Finding 2.6):** Change `scores[i] = (stm == White) ? ev : -ev` to `scores[i] = -ev`. Verify against eval convention; impacts training data quality.

5. **[High, ~1h] Fix null move `searchStack_[ply]` overwrite (Finding 1.3):** Store null-move hash at `searchStack_[ply+1]` instead. Prevents false repetitions and missed repetitions near null-move nodes.

6. **[High, ~1h] Add NNUE `incrementalUpdate` caller assertions (Findings 2.1, 2.2):** Add debug assertions that guard against promotion/en passant in the incremental path. Low effort; prevents future silent corruption.

7. **[High, ~1h] Fix repetition scan step size (Finding 1.4):** Change `i--` to `i -= 2` in the search-stack repetition scan. Halves work and eliminates cross-side false positives.

8. **[High, ~3h] Fix `setoption` race during active search (Finding 3.3):** Apply stop+join guard in the `setoption` handler, especially before `resizeTT()`. Prevents potential heap corruption.

9. **[High, ~4h] Add runtime AVX2 detection (Finding 4.13):** Add a startup CPUID check with a clear error message. Low build complexity; prevents silent crashes on incompatible CPUs.

10. **[High, ~2h] Fix self-play output validation (Finding 4.4):** Add file-size sanity check in the pipeline before passing the output to training. Prevents silent training quality degradation.

11. **[Medium, ~2h] Fix `updateStatus` to use `GameLogic::classify` (Findings 3.9, 3.10):** Eliminates the duplicate game-over logic and adds the missing insufficient-material draw detection in the GUI.

12. **[Medium, ~1h] Fix `--extra-data` argument validation (Finding 4.1):** Add pair-validation and type-checking of ratio values with clear error messages.

13. **[Medium, ~1h] Fix `fastMode` animation inconsistency (Finding 3.12):** One-line fix: add `|| botVsNNUE_` to the condition.

14. **[Medium, ~2h] Fix training phase continuous vs discrete mismatch (Finding 2.14):** Store actual phase count in training records and compute `p = phase/24.0` during training.

15. **[Medium, ~3h] Fix SWA validation gap (Finding 4.5):** Add a validation pass on the SWA model before export; only deploy if better than best non-SWA checkpoint.

16. **[Medium, ~1h] Fix `go` with no parameters defaulting to 5s (Finding 3.3):** Treat bare `go` as `go infinite` per UCI spec.

17. **[Medium, ~4h] Improve `SimpleSearcher` performance (Findings 2.3, 2.4):** Switch to incremental NNUE updates and add aspiration windows. Significant training-time speedup for ELO estimation.

18. **[Medium, ~2h] Fix checkpoint state consistency for early stopping (Finding 4.2):** Consolidate checkpoint saving to a single location after all counters are updated.

19. **[Low, ~30min] Fix `orderDuckPlacements` unsigned underflow (Finding 1.14):** Add `if (placements.count <= 1) return;` guard.

20. **[Low, ~1h] Fix PGO build profile data safety (Finding 4.17):** Move `llvm-profdata` output outside the build directory before `rm -rf`.

21. **[Low, ~1h] Add perft/smoke test `--no-wait` flag (Finding 3.24):** Enable CI/CD pipelines to run tests non-interactively.

22. **[Low, ~2h] Add quantisation error reporting to `save_weights_cpp` (Finding 4.6):** Log per-layer max error and zero-weight fraction after each export.

23. **[Info, ~3h] Add `ARCHITECTURE.md` documentation:** Cover NNUE update contracts, binary data format, TT design, and thread model.

24. **[Info, ~8h] Add NNUE consistency test and training data roundtrip test (Finding 4.20):** Foundation for regression testing of search correctness and training pipeline integrity.

---

## Strengths

Despite the findings above, the codebase demonstrates significant engineering sophistication:

- **Bitboard move generation:** The move generator correctly uses magic bitboards for slider attacks, with properly structured attack table lookups. The bitboard abstraction layer is clean and consistently used throughout move generation.

- **NNUE architecture and quantisation:** The network supports both float and int16-quantised inference paths with AVX2 acceleration. The quantised forward pass with `SCReLU` activation and three-phase blending is well-implemented. The incremental update logic is correct for the common case and fast.

- **Search algorithm breadth:** The engine implements a comprehensive set of modern alpha-beta enhancements: iterative deepening, aspiration windows, LMR, NMP, singular extensions, history heuristics, countermoves, multi-PV, duck chess support, and a proper TT replacement scheme. This is a substantial body of search knowledge correctly implemented.

- **Self-play and training pipeline:** The end-to-end pipeline (self-play generation → NNUE training → ELO evaluation → weight promotion) is architecturally sound and highly automated. The PowerShell pipeline script handles multi-generation training loops, early stopping, ETA estimation, and checkpoint management in a single cohesive workflow.

- **SFML-based GUI:** The visual interface supports human vs engine, engine vs engine, and NNUE training visualisation in real time. The animation and promotion dialog implementations are complete and functional.

- **PGN and FEN parsing:** The PGN-to-training-data converter and FEN parser appear comprehensive for standard use cases.

- **Test harness:** The perft results correctly validate move generation (the gold standard for chess engine correctness), and the smoke test suite covers a useful range of tactical and positional scenarios.

---

*End of Report — 92 findings across 4 sections*
