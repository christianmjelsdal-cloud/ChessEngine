# ChessEngine — Comprehensive Code Audit

**Date:** 2026-03-29  
**Auditor:** Automated Code Audit (AI-assisted)  
**Scope:** Full codebase (~30K lines C++/Python) — engine, NNUE, training, GUI, build system

---

## Executive Summary

ChessEngine is a C++17 chess engine featuring NNUE evaluation (768→256→1 HalfKAv2 architecture with SCReLU activations and WDL output), Lazy SMP parallel search, UCI protocol support, an optional SFML-based GUI, and a complete training pipeline including self-play generation, a Windows-native training orchestrator (TrainingRunner), and Python training scripts.

The codebase is **generally well-engineered** with evidence of extensive iterative improvement (many `AUDIT FIX` and `FIX` annotations throughout). Key strengths include proper bitboard + mailbox dual representation, magic bitboards for sliding pieces, thorough Zobrist hashing, modern search techniques (PVS, LMR, null-move pruning, singular extensions), AVX2 SIMD throughout the NNUE inference pipeline, and a well-structured training data format with versioned headers.

**However, 122 findings were identified**, including 1 critical bug, 3 high-severity issues, 27 medium-severity issues, and numerous low/informational items. The most pressing concerns are a debug-crashing assert in `VisualGame.cpp`, silent accumulator corruption for en passant in NNUE inference, and test/production binary format divergence.

---

## Severity Summary Table

| Severity | Count | Description |
|----------|-------|-------------|
| **Critical** | 1 | Crashes or data corruption in normal operation |
| **High** | 3 | Incorrect behavior under specific conditions |
| **Medium** | 27 | Bugs with workarounds, significant code quality issues, architecture problems |
| **Low** | 38 | Minor bugs, code quality, style issues |
| **Info** | 23 | Observations, positive notes, documentation suggestions |
| **Total** | **92** | |

---

## Critical & High Severity Findings

### C-1. [CRITICAL] Assert always fires in `startEngineThinking()` — crashes every debug build

- **File:** `VisualGame.cpp`, line 1133
- **Category:** Bug
- **Description:** Line 1107 sets `engineThinking = true`, then line 1133 asserts `!engineThinking`. This assert will **always fire** in debug builds, crashing the program every time `startEngineThinking()` is called.
- **Fix:** Move the assert to before line 1107 (before `engineThinking = true` is set), or remove it if the guard at line 1105 is sufficient.

---

### H-1. [HIGH] En passant incremental accumulator update silently corrupts NNUE values

- **File:** `NNUE.cpp`, lines 200–208
- **Category:** Correctness
- **Description:** The `incrementalUpdate` function has an `assert`-only check for en passant but no runtime handling. In release builds (NDEBUG defined), en passant moves will silently corrupt the accumulator because the captured pawn is not on the destination square — it's on a different rank. The code assumes the captured piece is at `toSq`. Callers must know to call `refreshAccumulator` instead, which is fragile and undocumented at the call site.
- **Fix:** Add a **runtime check** (not just assert) that detects en passant and calls `refreshAccumulator`, similar to how king moves trigger a refresh at line 210. Alternatively, handle the EP capture explicitly by removing the correct pawn feature.

---

### H-2. [HIGH] Threefold repetition check scans entire history — false positives possible

- **File:** `VisualGame.cpp`, lines 1280–1291
- **Category:** Correctness
- **Description:** `VisualGame::updateStatus()` scans the **entire** `positionHistory_` for repetition, while `GameLogic::isThreefoldRepetition()` correctly limits the scan to the last `halfMoveClock` entries (repetitions can't cross irreversible moves). The VisualGame version could produce false positives in rare cases where the same Zobrist hash appears across a pawn move/capture boundary.
- **Fix:** Replace the manual loop with a call to `GameLogic::isThreefoldRepetition(positionHistory_, board.halfMoveClock)`. The TODO at line 1256 already suggests this consolidation.

---

### H-3. [HIGH] Test binary format incompatibility — `test_pipeline.py` reads wrong offset

- **File:** `test_pipeline.py`, lines 121–122
- **Category:** Bug
- **Description:** `test_bin_format()` starts reading position data at offset=4 (old 4-byte header), but files are written using `training_format.write_header()` which uses a 9-byte versioned header (4 magic + 1 version + 4 count). This means parsing starts at the wrong byte boundary, causing all position records to be read incorrectly.
- **Fix:** Use `training_format.read_header()` to get the data offset dynamically, or use `HEADER_SIZE` from `training_format.py`.

---

### H-4. [HIGH] `test_training_integration.py` writes incompatible binary format

- **File:** `test_training_integration.py`, lines 33–43
- **Category:** Correctness
- **Description:** `create_tiny_dataset()` writes positions as fixed-size records with separate `white[]`/`black[]` int32 arrays — a completely different format from the sparse variable-length format used by all production code. This test doesn't validate the actual production binary format.
- **Fix:** Use the same format as `create_fake_bin()` from `test_pipeline.py`, or create a shared test utility that writes the production binary format.

---

## Medium Severity Findings

### Architecture & Code Duplication

**M-1. `applyMove` duplicates nearly all logic from `makeMove`** — `Board.cpp:151–275`  
The two methods contain ~90% identical logic for en passant, castling, piece movement, promotion, etc. Any fix to one must be mirrored in the other. Extract shared logic into a private helper.

**M-2. `VisualGame.cpp` is a 2000+ line monolith** — `VisualGame.cpp:1`  
Mixes rendering (~700 lines), input handling (~300), engine thread management (~200), training orchestration (~200), and game logic (~200). The comment at line 1 already acknowledges this. Split into `VisualGameRender.cpp`, `VisualGameInput.cpp`, etc.

**M-3. Duplicated game-over detection between `updateStatus()` and `GameLogic::classify()`** — `VisualGame.cpp:1256–1343`  
The TODO at line 1256 acknowledges this. Replace with a single call to `GameLogic::classify()`.

**M-4. ~240 lines of duplicated Python match scripts in TR_Pipeline.cpp** — `TR_Pipeline.cpp:523–651, 799–922`  
`elo_match.py` and `swa_match.py` are nearly identical embedded Python scripts differing only in variable naming. Extract into a single parameterized script.

**M-5. Engine class is a god object (~3MB+ per instance)** — `Engine.h:37–394`  
Handles search, evaluation, NNUE, TT, move ordering, time control, pondering, Lazy SMP, MultiPV, Syzygy, and UCI output. Consider extracting `TimeManager`, `MoveOrderer`, `TranspositionTable`, and `SearchStack` classes.

**M-6. Training-only code linked into engine binary** — `CMakeLists.txt:44–55`  
`SelfPlayGen.cpp` and `NNUETrainer.cpp` are included in `chess_core` but only needed for training. Split into `chess_core` and `chess_training` libraries.

### Correctness Bugs

**M-7. `applyMove` doesn't update `nonPawnMaterial`/phase** — `Board.cpp:274`  
`applyMove` sets `hash=0` and calls `syncBitboards()` but leaves `nonPawnMaterial`/phase stale. If used in Duck Chess search, evaluation scores may be incorrect.

**M-8. `applyMove` uses O(64) `syncBitboards()` instead of incremental update** — `Board.cpp:273`  
`makeMove` does efficient incremental bitboard updates; `applyMove` rescans all 64 squares. Performance problem if used in search paths.

**M-9. Fallback move parsing without 'moves' keyword doesn't validate legality** — `UCI.cpp:236–248`  
When the `moves` keyword is omitted, any 4-character coordinate-like string could be applied as a move, potentially corrupting board state.

**M-10. `pieceBB[0]` (PieceType::None) unguarded** — `Board.h:38–44`  
The comment warns it "MUST NEVER be written to," but `removeBitboard`/`addBitboard` have no guard against `PieceType::None`. Add `assert(pt != PieceType::None)`.

**M-11. `squareBB` has no bounds check — UB with invalid Square** — `Bitboard.h:128–140`  
`1ULL << sq` with `sq = -9` (from invalid `Square{-1,-1}`) is undefined behavior. Add `assert(sq >= 0 && sq < 64)`.

**M-12. `genSummary` logs stale zero generation duration** — `TR_Pipeline.cpp:1103–1108`  
`prevGenCompletedSec` is updated to `lastGenCompletedSec` *before* the delta is computed, so `genSec` is always ~0. Move the logging before the update.

**M-13. Cooldown loop counter double-decremented** — `TR_Benchmark.cpp:475–477`  
Both the for-loop and an explicit `cd--` decrement the counter, halving the intended 30-second cooldown to ~15 seconds.

**M-14. `pieceCount()` in Syzygy.cpp uses undefined `popcount()` on GCC/Clang** — `Syzygy.cpp:55`  
The file-local `popcount()` is only defined for MSVC and non-GCC/Clang fallback. On GCC/Clang without `HAS_SYZYGY`, it fails to compile. Use `BB::popcount()`.

**M-15. `ponderSoftMs_`/`ponderHardMs_` are non-atomic but cross-thread** — `UCI.h:65–66`  
These are plain `int` but potentially accessed from different threads. Make atomic for safety.

### NNUE & Training

**M-16. No runtime SIMD detection — hard AVX2+FMA dependency** — `NNUE.cpp:12–17`  
No CPUID check or fallback. Crashes with SIGILL on CPUs without AVX2+FMA. Add a startup check.

**M-17. Dead code after except block in train_nnue.py** — `train_nnue.py:2919–2937`  
Post-training SWA and summary code is unreachable because the try block returns and the except calls `sys.exit(1)`. Move SWA application before the return.

**M-18. `generate()` returns `int` but `totalPositions` is `uint64_t`** — `SelfPlayGen.cpp:1051`  
Truncation on large runs generating >2B positions. Return `int64_t`.

**M-19. Softmax move selection duplicated 3× in `playGame()`** — `SelfPlayGen.cpp:370–507`  
Identical pattern for opening, post-opening, and epsilon-greedy phases. Extract a helper function.

**M-20. Feature preconversion uses per-position heap vectors** — `NNUETrainer.cpp:1100–1130`  
Creates massive heap fragmentation for millions of positions. Use flat buffers with offset arrays.

**M-21. Single-threaded C++ training loop** — `NNUETrainer.cpp:1128–1520`  
Each position's forward/backward is independent within a batch — ripe for parallelization.

### Python & Build

**M-22. Duplicated phase classification across 3 Python files** — `analyze_phases.py:22–39`  
`compute_material_phase()` is copy-pasted in `analyze_phases.py`, `generate_draws.py`, and the C++ engine. Centralize in `training_format.py`.

**M-23. Global mutable `_draw_source_stats` never reset** — `pgn_to_training.py:136`  
Stats bleed across multiple calls. Make it a return value.

**M-24. Auto-installing pip packages at runtime** — `test_one_game.py:76–82`  
Security risk in CI. Use pinned `requirements.txt`.

**M-25. `SmokeTest.cpp` blocks on stdin in CI** — `SmokeTest.cpp:254–255`  
Has `std::cin.get()` without a `CI_BUILD` guard (unlike `Test.cpp`). Add the guard.

### Training Runner

**M-26. `ResolvePythonCmd` uses `system()` — PATH manipulation risk** — `TR_Pipeline.cpp:13–14`  
Invokes the shell for Python detection. Use `CreateProcessW` with explicit search.

**M-27. `TerminateThread` fallback risks process corruption** — `TrainingRunner.cpp:770–782`  
Doesn't release locks or run destructors. The 10s timeout mitigates this, but consider cooperative shutdown.

---

## Low Severity & Informational Highlights

### Recurring Themes

**Code Duplication (8 findings):** Cancel-and-join patterns in VisualGame key handlers, duck placement logic between mouseDown/mouseUp, eval bar rendering, feature encoding across Python scripts. The codebase would benefit from a systematic deduplication pass.

**Memory & Performance (6 findings):** History/countermove tables consume ~1.5MB per Engine (×N for Lazy SMP), `refreshAccumulatorQ` iterates all 40960 features instead of the ~30 active ones, `getLegalCaptures` generates all moves then filters, `exeDir()` is called repeatedly without caching.

**Undefined Behavior Risks (4 findings):** `lsb()`/`msb()` with b=0 are UB in release builds, `isSquareAttacked` vs `attackersTo` have inconsistent duck handling, Zobrist tables may diverge across TUs with different `DUCK_CHESS` definitions.

**Test Quality (4 findings):** Hardcoded offsets, incompatible binary formats in tests, fragile Args class mirroring train_nnue.py's argparse, debug perft is orders of magnitude slower due to `verifyApplyMakeEquivalence`.

**Minor Bugs (5 findings):** `addBitboard` doesn't assert target square is empty, `isSquareAttacked` is legacy dead code, WMI `VT_UI4` reads signed `intVal`, ETA parsing may misinterpret MM:SS as HH:MM, stdin fd leak on Windows.

### Notable Positive Observations

- **Zobrist hashing** is correctly implemented with `std::call_once` initialization
- **training_format.py** provides excellent format centralization with versioned headers
- **MT19937 seed decorrelation** using xorshift32 mixing is well done
- **Self-play adjudication** methods are comprehensive (material, tablebase, move limit)
- **Perft test suite** with 26 cases provides strong move generation validation
- **Duck Chess** variant is cleanly implemented via conditional compilation

---

## Architecture Overview & Assessment

### Strengths

1. **Dual representation (bitboard + mailbox)** with synchronized updates provides both fast attack computation and easy piece access
2. **NNUE architecture** is modern (HalfKAv2 with king buckets, SCReLU, quantized INT16 accumulators, WDL output head)
3. **Incremental accumulator updates** in search avoid expensive full recomputation
4. **Well-structured training pipeline** from self-play through training to Elo validation
5. **Extensive prior audit fixes** show commitment to code quality
6. **Good separation of concerns** in most areas (GameLogic extracted, training_format shared)

### Weaknesses

1. **Engine as god object** — The `Engine` class handles too many responsibilities (~3MB per instance, 50+ member variables)
2. **VisualGame monolith** — 2000+ lines mixing 5+ concerns
3. **Two move application paths** — `applyMove` and `makeMove` are dangerously similar but subtly different
4. **Hard SIMD dependency** — No fallback for non-AVX2 CPUs
5. **Test/production format divergence** — Some tests write different binary formats than production code
6. **Training code in engine binary** — Increases binary size unnecessarily

### Code Statistics (estimated)

| Component | Lines | Files |
|-----------|-------|-------|
| Core Engine (Board, MoveGen, Bitboard, Types) | ~5,000 | 8 |
| Search Engine | ~3,500 | 2 |
| NNUE Inference + Training (C++) | ~4,000 | 4 |
| UCI + GameLogic | ~1,500 | 4 |
| GUI (VisualGame) | ~2,000 | 2 |
| Self-Play Generator | ~1,100 | 2 |
| Training Runner (Win32 GUI) | ~5,500 | 8 |
| Python Scripts | ~5,000 | 10 |
| Tests + Build | ~2,000 | 5 |
| **Total** | **~30,000** | **~45** |

---

## Recommendations

### Top 10 Prioritized Action Items

1. **Fix C-1: Assert in `startEngineThinking()`** — Crashes every debug build. 5-minute fix.

2. **Fix H-1: En passant accumulator corruption** — Add runtime EP detection in `incrementalUpdate`. This silently degrades NNUE evaluation quality for EP positions in release builds.

3. **Fix H-3 & H-4: Test binary format alignment** — Tests don't validate production format. Update `test_pipeline.py` to use `HEADER_SIZE` and rewrite `test_training_integration.py` to use the production binary format.

4. **Fix M-12: genSummary stale duration** — Move `logGenSummary` before updating `prevGenCompletedSec`. 2-line swap.

5. **Consolidate game-over detection (M-3)** — Replace `VisualGame::updateStatus()` reimplementation with `GameLogic::classify()`. Fixes H-2 as a side effect.

6. **Add NNUE SIMD runtime check (M-16)** — Add a startup CPUID check in `NNUE::Network` constructor. Provide a clear error message instead of SIGILL.

7. **Extract shared move logic (M-1)** — Refactor `applyMove` to delegate to `makeMove` with a discarded UndoInfo, or extract common logic into a private helper. Fixes M-7 and M-8 as side effects.

8. **Deduplicate Python match scripts (M-4)** — Extract the ~240 lines of embedded Python into a single parameterized `uci_match.py` template.

9. **Split Engine class (M-5)** — Extract `TimeManager`, `MoveOrderer`, and `TranspositionTable` to reduce the god object. This is a larger refactor but pays dividends for testability and memory efficiency.

10. **Centralize Python feature encoding (M-22)** — Move phase classification and feature index computation into `training_format.py` and import everywhere.

---

## Updated Files Assessment

Five files were provided as updated/fixed versions: **TR_Benchmark.cpp**, **TR_Fwd.h**, **TR_Pipeline.cpp**, **TR_Types.h**, and **TrainingRunner.cpp**.

### What the updates fixed
- Proper COM lifecycle management and atomic flags for cross-thread communication
- Pause-aware timing in the training pipeline
- Path injection validation for security
- Structured file logging (`TR_Logger.h` integration)
- Config validation before pipeline start
- Various prior audit fix annotations throughout

### What remains
- **M-12:** `genSummary` still logs stale zero-duration values (timing bug in the update order)
- **M-13:** Cooldown loop double-decrement halves intended cooldown time
- **M-4:** Duplicated Python match scripts (~240 lines each, nearly identical)
- **M-26:** `ResolvePythonCmd` still uses `system()` instead of `CreateProcessW`
- **M-27:** `TerminateThread` fallback on shutdown is risky
- Several low-severity items: static arrays in headers (ODR), `pushLog` inline dependencies, redundant lock acquisitions, ETA parsing ambiguity

The updated Training Runner files are substantially improved over typical patterns but still have the medium-severity bugs listed above that should be addressed.

---

*End of audit report.*
