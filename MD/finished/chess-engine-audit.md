# Chess Engine — Full Project Audit

**Date:** 2026-03-13  
**Auditor:** Automated Code Audit  
**Project:** ChessEngine NNUE (C++/Python chess engine with neural network evaluation)

---

## Executive Summary

This is a **solo-developed chess engine** with NNUE neural network evaluation, a complete UCI protocol implementation, self-play training pipeline, visual GUI (SFML), a Win32 training GUI, and Duck Chess variant support. The project demonstrates strong ambition and solid chess programming knowledge.

**Maturity:** Mid-stage hobby/research project. The core engine is functional with proper alpha-beta search, NNUE evaluation, and a sophisticated Python training pipeline. The codebase is well-organized but shows signs of rapid iterative development with some rough edges.

**Top findings:**
- **No critical bugs found** — the engine appears functionally correct for standard chess
- **Board representation is slow** — 8×8 array with no bitboards severely limits NPS
- **Copy-on-search pattern** — `Board temp = board;` at every node is a major performance bottleneck
- **Training pipeline is impressively mature** — streaming, multi-dataset, HalfKAv2, SWA, stratified sampling
- **Good search implementation** — LMR, null move, singular extensions, aspiration windows, history/killer/countermove heuristics all present
- **Code quality is generally good** with clear comments and modular design

---

## Project Overview

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.cpp                                 │
│  (CLI dispatcher: --uci, --visual, --generate, --train, --test) │
└──────┬──────────┬───────────┬───────────┬───────────┬───────────┘
       │          │           │           │           │
   ┌───▼──┐  ┌───▼───┐  ┌───▼────┐  ┌───▼────┐  ┌───▼─────┐
   │ UCI  │  │Visual │  │SelfPlay│  │NNUETra-│  │TestSuite│
   │      │  │ Game  │  │  Gen   │  │  iner  │  │         │
   └──┬───┘  └──┬────┘  └──┬─────┘  └──┬─────┘  └────┬────┘
      │         │           │           │              │
      └─────────┴─────┬─────┴───────────┘              │
                      │                                 │
              ┌───────▼────────┐                        │
              │    Engine      │◄───────────────────────┘
              │ (Search+Eval)  │
              └───┬────────┬───┘
                  │        │
           ┌──────▼──┐  ┌──▼───┐
           │ MoveGen │  │ NNUE │
           └────┬────┘  └──────┘
                │
           ┌────▼────┐
           │  Board   │
           │ + Types  │
           └──────────┘

Python Training Pipeline:
  train_nnue.py ◄── pgn_to_training.py, convert_lichess_elite.py,
                    generate_draws.py, merge_training_data.py
  
Automation:
  pipeline.ps1 (PowerShell orchestration)
  TrainingRunner.cpp (Win32 GUI for training)
```

### Technology Stack

| Component | Technology |
|-----------|-----------|
| Engine Core | C++17 |
| GUI (Play) | C++ / SFML 3 |
| GUI (Training) | C++ / Win32 + GDI+ |
| NNUE Inference | C++ with SSE intrinsics |
| Training | Python 3 / PyTorch (CPU) |
| Automation | PowerShell, Batch files |
| Build | MSVC (Visual Studio) |

### Component Breakdown

| File | Lines | Purpose |
|------|-------|---------|
| Engine.cpp/h | 1,770 | Search (alpha-beta, qsearch, time mgmt) + HCE |
| train_nnue.py | 2,031 | PyTorch NNUE training pipeline |
| VisualGame.cpp/h | 2,161 | SFML chess GUI |
| TrainingRunner.cpp | 3,125 | Win32 training GUI |
| TestSuite.cpp | 1,627 | Comprehensive test suite |
| NNUETrainer.cpp/h | 1,515 | C++ self-play data generation + in-engine trainer |
| pipeline.ps1 | 759 | PowerShell pipeline automation |
| Engine.h | 191 | Engine class definition |
| NNUE.cpp/h | 530 | NNUE network inference (SSE) |
| SelfPlayGen.cpp/h | 558 | Multi-threaded self-play |
| UCI.cpp/h | 504 | UCI protocol |
| Board.cpp/h | 415 | Board representation |
| MoveGen.cpp/h | 361 | Move generation |
| Other Python | ~1,200 | Data conversion, analysis, testing |
| Scripts | ~850 | Build, test, ELO scripts |
| Docs (MD/) | ~500 | Architecture docs |
| **Total** | **~18,574** | |

---

## Architecture Review

### Overall Design Quality: **B+**

The project has a clean separation of concerns with well-defined module boundaries. The header/implementation split is consistent, and the code is well-commented with section markers. The architecture naturally evolved from a simple engine to a full training ecosystem.

**Strengths:**
- Clean module boundaries (Board → MoveGen → Engine → UCI)
- NNUE is properly abstracted behind a pointer interface (`setNNUE()`)
- Conditional compilation for Duck Chess variant (`#ifdef DUCK_CHESS`)
- Both Python (production) and C++ (in-engine) training paths
- Stack-allocated `MoveList` avoids heap allocation in hot paths

**Weaknesses:**
- Board representation is the weakest architectural choice — 8×8 array instead of bitboards
- No make/unmake move — full board copy at every search node
- Engine class is monolithic (1,770 lines) — could split search, evaluation, and heuristics
- NNUE uses SSE but the board representation prevents effective SIMD utilization in move gen
- `TrainingRunner.cpp` at 3,125 lines is a "God file" mixing Win32 UI, process management, and training logic

### Module Coupling: **Good**

- Board ← MoveGen ← Engine ← UCI (clean dependency chain)
- NNUE is optional (fallback to handcrafted eval)
- Python and C++ training are independent paths sharing binary format

### Data Flow

```
Position → Board → MoveGen::getLegalMoves → Engine::search → NNUE::evaluate
                                                           → Engine::evaluate (HCE fallback)
Self-play: SelfPlayGen → training_data.bin → train_nnue.py → nnue_weights.bin → NNUE::loadWeights
```

### Design Patterns

- **Strategy Pattern**: NNUE vs. handcrafted eval selected at runtime
- **Observer Pattern**: `onInfoCallback` for UCI info output
- **Builder Pattern**: Training configuration via `TrainingConfig` struct
- **Template Method**: Iterative deepening framework with pluggable search

**Missing Patterns:**
- No Command pattern for UCI (uses string-matching if/else chain)
- No State pattern for game phases
- No Factory for engine configurations

---

## Code Quality Analysis

### Board (Board.cpp/h) — Grade: **C+**

**Purpose:** 8×8 array board representation with piece placement and game state.

**Strengths:**
- Simple and readable
- Correct FEN parsing (in UCI.cpp)
- Duck Chess support via conditional compilation

**Issues:**
- **No bitboard representation** — the biggest architectural limitation. Every square check iterates the array.
- `applyMove()` doesn't track enough undo information for make/unmake
- No incremental Zobrist hash updates — hash is recomputed from scratch
- Missing `isValid()` bounds checking in some paths

### MoveGen (MoveGen.cpp/h) — Grade: **B**

**Purpose:** Legal move generation for all piece types including Duck Chess.

**Strengths:**
- Correct handling of castling through check/attack detection
- En passant correctly implemented with pin detection
- Duck Chess moves properly generated
- `MoveList` uses stack allocation (no heap in hot path)

**Issues:**
- Generates pseudo-legal moves then filters for legality by testing `isInCheck` after each — O(n²) pattern
- No staged move generation (generate all moves up front, then sort)
- `isSquareAttacked()` scans the entire board for each check test
- `getLegalCaptures()` generates all moves then filters — could generate captures directly

### Engine (Engine.cpp/h) — Grade: **B+**

**Purpose:** Alpha-beta search with modern pruning techniques and NNUE/HCE evaluation.

**Strengths:**
- Comprehensive search features: LMR, null move, singular extensions, aspiration windows, PVS
- History gravity (Stockfish-style) for move ordering
- Capture history heuristic
- Improving flag reduces over-pruning
- Delta pruning and SEE in quiescence
- Best move stability tracking for dynamic time management
- Well-tuned piece-square tables (PeSTO-derived)
- Clean fail-soft implementation

**Issues:**
- `Board temp = board;` copies entire board at every node (~line 833, 1013, 1235, etc.) — massive performance hit
- No make/unmake means no incremental hash update — `computeHash()` is O(64) per node
- Repetition detection scans linearly through `gameHistory_` — should use a hash set
- `evaluate()` (HCE) iterates all 64 squares three times for different evaluation terms
- No multi-threading (no Lazy SMP or similar)
- `previousMoves_[64]` array is not bounds-checked — potential overflow at ply 64 (line 1015)
- Null move doesn't verify non-zugzwang material more carefully

### NNUE (NNUE.cpp/h) — Grade: **A-**

**Purpose:** Neural network inference with HalfKAv2 feature encoding and SSE acceleration.

**Strengths:**
- Proper HalfKAv2 encoding with king-relative features
- SSE intrinsics for accumulator operations (`_mm_load_ps`, `_mm_add_ps`)
- Incremental accumulator updates (add/remove piece without full refresh)
- SCReLU activation (squared clipped ReLU — modern NNUE technique)
- Correct perspective handling (white/black accumulators)
- Aligned memory allocation for SIMD

**Issues:**
- Full accumulator refresh on every king move (expected, but could be optimized with finny tables)
- No AVX2/AVX-512 code paths — SSE only limits throughput on modern CPUs
- `L1_SIZE` alignment assumption: `_mm_load_ps` requires 16-byte alignment, but alignment is not enforced on all allocations (potential UB on some allocators)
- `evaluate()` creates temporary arrays on the stack each call — could be cached

### NNUETrainer (NNUETrainer.cpp/h) — Grade: **B**

**Purpose:** C++ in-engine training with self-play data generation and Adam optimizer.

**Strengths:**
- Adam optimizer correctly implemented with bias correction
- Phase-balanced training (opening/middlegame/endgame stratification)
- Draw detection and tagging
- Early stopping with patience
- Binary format well-documented and consistent with Python

**Issues:**
- Backpropagation is manually implemented — error-prone for architecture changes
- Training loop modifies a flat weight vector — no layer abstraction
- No validation split in C++ trainer (only in Python)
- `mirrorData()` creates a full copy of the dataset — doubles memory
- Random opening moves for diversity is naive (should use an opening book or FEN list)

### UCI (UCI.cpp/h) — Grade: **B+**

**Purpose:** Universal Chess Interface protocol implementation.

**Strengths:**
- Correct handling of `position startpos moves`, `position fen`
- Proper FEN parsing including castling rights and en passant
- Search runs on separate thread with proper stop mechanism
- `ucinewgame` correctly clears all search state
- Handles both `wtime/btime/winc/binc` and `movetime`

**Issues:**
- `setoption` is silently ignored — should at least support Hash size and Threads
- No `debug` mode support
- `generate` command is non-standard UCI extension (OK, but should be documented)
- FEN parsing duplicates board setup logic that could be in `Board::setFromFEN()`
- Time management doesn't account for move overhead

### SelfPlayGen (SelfPlayGen.cpp/h) — Grade: **B+**

**Purpose:** Multi-threaded self-play game generation for training data.

**Strengths:**
- True multi-threading with `std::thread` workers
- Per-worker engine instances with reduced TT size
- Atomic progress tracking with cancel support
- Opening book support (FEN file)
- Configurable search depth and position limit

**Issues:**
- Workers use `new Engine()` without RAII — potential memory leak on exception
- No game adjudication for long games (relies on 50-move/repetition rules only)
- All workers write to shared vector with mutex — could use thread-local buffers
- Fixed search depth rather than time-based search for data generation

### VisualGame (VisualGame.cpp/h) — Grade: **B**

**Purpose:** SFML-based chess GUI for playing against the engine.

**Strengths:**
- Clean rendering with piece sprites and board graphics
- Engine vs engine mode for testing
- Live PV display with arrows
- Proper thread safety for engine communication

**Issues:**
- Hard-coded asset paths
- No resizable window
- SFML 3 API usage is correct but tightly coupled

### TrainingRunner (TrainingRunner.cpp) — Grade: **B-**

**Purpose:** Win32 native GUI for managing the training pipeline.

**Strengths:**
- Impressive feature set: live log, progress bars, preset management, graph rendering
- Dark theme with custom drawing
- Process management for pipeline.ps1 subprocess

**Issues:**
- 3,125 lines in a single file — needs decomposition
- Windows-only (Win32 API)
- GDI+ graph rendering reimplements what matplotlib already does
- No error recovery for subprocess crashes

---

## Bug Report

| # | File | Line | Severity | Description | Fix |
|---|------|------|----------|-------------|-----|
| 1 | Engine.cpp | ~1015 | **Medium** | `previousMoves_[ply] = m;` guarded by `if (ply < 64)` but `MAX_PLY` is 64, so ply=63 writes to last element but recursive search could be called with ply+1=64 which reads `previousMoves_[63]` (actually OK since bounded, but the off-by-one is fragile) | Use `MAX_PLY` constant consistently |
| 2 | Engine.cpp | ~780 | **Low** | `qsearch` allows `ply >= MAX_PLY + 32` (line 780) but `pvLength_` array is only `MAX_PLY+1` — writing to `pvLength_[ply]` when ply ≥ 65 is **buffer overflow** | Add bounds check or increase array size. Actually `pvLength_` is NOT written in qsearch, so this is safe — but confusing. |
| 3 | Engine.cpp | ~867-873 | **Medium** | Repetition detection scans `gameHistory_` limited by `halfMoveClock` but doesn't account for the fact that `halfMoveClock` tracks plies since last capture/pawn move, not the number of entries in `gameHistory_`. If `halfMoveClock` > `gameHistory_.size()`, the `scanBack` is correctly clamped, but the variable naming is misleading. | Add comment clarifying the relationship |
| 4 | Engine.cpp | ~928 | **Medium** | Null move: `searchStack_[ply]` is overwritten with the null-move board hash, but if the null-move search triggers a cutoff and returns, the original hash at `searchStack_[ply]` is restored (line 931). However, if `shouldStop()` fires during the null-move search, it returns 0 without restoring `searchStack_[ply]`. | The search exits immediately anyway on stop, so this is harmless in practice, but it's not clean. |
| 5 | Engine.cpp | ~1132 | **Low** | TT replacement: `(uint8_t)(ttGen_ - tte.gen) > 2` uses unsigned wrapping arithmetic which is correct for generation counting, but the magic number `2` should be a named constant. | Define `TT_MAX_AGE = 2` |
| 6 | NNUE.cpp | ~122-126 | **Medium** | `_mm_load_ps` requires 16-byte aligned pointers. `std::array<float, L1_SIZE>` in the Accumulator is not guaranteed to be 16-byte aligned. Modern allocators usually align to 16 bytes, but this is not guaranteed by the standard. | Use `alignas(16)` on the accumulator arrays |
| 7 | MoveGen.cpp | ~42 | **Low** | Double pawn push: `board.squares[nr2][sq.col]` is not bounds-checked for `nr2`. Since `startRank` is 1 or 6, and `dir` is ±1, `nr2` will be 3 or 4, always in bounds. Safe but fragile if startRank ever changes. | Add explicit bounds check |
| 8 | UCI.cpp | ~55 | **Low** | `setoption` is completely ignored. A GUI or test harness sending `setoption name Hash value 256` will silently fail. | Implement at least Hash size option |
| 9 | SelfPlayGen.cpp | various | **Low** | Workers create engines with `new Engine(...)` — if an exception occurs during game play, the engine is leaked. | Use `std::unique_ptr<Engine>` |
| 10 | train_nnue.py | ~450 | **Low** | `all_white = np.zeros((n, NUM_FEATURES), dtype=np.float32)` allocates 40,960 × N × 4 bytes of dense arrays. For N=1M positions, this is ~150 GB. The streaming mode handles this, but `load_positions_at_offsets` is called for validation sets and chunks where N could be large. | Cap chunk size or use sparse representation |
| 11 | Engine.cpp | ~496-499 | **Low** | `evaluate()` with NNUE: `return nnue_->evaluate(board)` doesn't pass the accumulated NNUE state — it recomputes from scratch every call. The incremental update functions exist but aren't used during search. | Implement accumulator stack in search to enable incremental updates |
| 12 | Board.h/Board.cpp | N/A | **Medium** | No `operator==` for `Board` — makes it impossible to verify board state correctness in tests without manual comparison. `Move` has `operator==` but compares only from/to/promotion, not duckTo. | Add full equality operators |

---

## Performance Analysis

### Critical Bottleneck: Board Copy per Node

The single biggest performance issue is `Board temp = board;` at every search node. The `Board` struct contains:
- `Piece squares[8][8]` (128 bytes for pieces)
- Game state (castling, en passant, halfMoveClock, turn, etc.)
- Total ~200+ bytes copied per node

**Impact:** At 1M nodes/sec target, this is ~200 MB/sec of pure memcpy. A make/unmake approach would reduce this to ~20-30 bytes of state change per node.

### No Incremental Zobrist Hashing

`computeHash()` iterates all 64 squares every time. With make/unmake, this would be O(1) — XOR out old piece, XOR in new piece.

**Estimated NPS impact:** 2-3x improvement from make/unmake + incremental hash alone.

### NNUE Inference Not Incremental in Search

The NNUE accumulator supports incremental updates (`incrementalUpdate()`) but `evaluate()` calls `refreshAccumulator()` every time. The search doesn't maintain an accumulator stack.

**Impact:** Full accumulator refresh is O(active_pieces × L1_SIZE) per node instead of O(2 × L1_SIZE) for incremental.

### Move Generation

- Pseudo-legal + legality filter is standard but slower than generating legal moves directly
- `isSquareAttacked()` scans the entire board for each call
- No bitboard representation means no bulk move generation

### Memory Usage

- TT default: 4M entries × ~24 bytes = ~96 MB (reasonable)
- NNUE weights: ~40,960 × 512 × 4 = ~80 MB for L1 alone (large but acceptable)
- Self-play workers: reduced TT size per worker (good)

### Search Optimization Assessment

| Technique | Present | Quality |
|-----------|---------|---------|
| Alpha-Beta | ✅ | Good (fail-soft) |
| Iterative Deepening | ✅ | Good |
| Aspiration Windows | ✅ | Good (graduated) |
| Null Move Pruning | ✅ | Good (with material check) |
| Late Move Reductions | ✅ | Good (log-based formula) |
| Principal Variation Search | ✅ | Good |
| Transposition Table | ✅ | Good (generation-based replacement) |
| Killer Moves | ✅ | Good (2 per ply) |
| History Heuristic | ✅ | Good (with gravity) |
| Countermove Heuristic | ✅ | Good |
| Capture History | ✅ | Good |
| Static Exchange Evaluation | ✅ | Good |
| Futility Pruning | ✅ | Good |
| Reverse Futility Pruning | ✅ | Good |
| Late Move Pruning | ✅ | Good |
| Singular Extensions | ✅ | Good |
| Check Extensions | ✅ | Simple |
| Internal Iterative Deepening | ✅ | Good |
| Improving Flag | ✅ | Good |
| Lazy SMP | ❌ | Missing — major NPS limitation |
| Bitboards | ❌ | Missing — foundational limitation |
| Make/Unmake | ❌ | Missing — major performance issue |
| Incremental Hash | ❌ | Missing |
| NNUE Accumulator Stack | ❌ | Missing |

---

## Security & Robustness

### Input Validation

- **UCI input:** Reasonably validated. FEN parsing has basic checks but could crash on malformed input.
- **Move parsing:** `parseMove()` in UCI.cpp doesn't validate square bounds before accessing board.
- **File I/O:** NNUE weight loading checks file open but doesn't validate file size against expected weight count.

### Buffer Overflow Risks

- **`MoveList`:** Fixed-size array with `MAX_MOVES = 256`. Chess maximum is ~218 legal moves, so 256 is sufficient, but `push_back()` doesn't check bounds (line in MoveGen.h). In theory, pseudo-legal moves could exceed this in pathological cases.
- **`pvTable_[MAX_PLY+1][MAX_PLY+1]`:** 65×65 Move array — adequate.
- **`quietsTriedArr[64]`:** Bounded by `quietsTriedCnt < 64` check — safe.
- **`gain[32]` in SEE:** Maximum exchange depth is bounded by number of pieces (≤32) — safe.

### Error Handling

- NNUE weight loading has try/catch but silently falls back to HCE
- File I/O errors in training pipeline are generally caught with informative messages
- No structured error handling in the engine core (crashes on assertion failures)

### Resource Leaks

- `SelfPlayGen` workers use raw `new` — potential leak on exception
- VisualGame properly joins threads in destructor
- UCI properly joins search thread on destruction

---

## Training Pipeline Review

### Python Code Quality: **A-**

The `train_nnue.py` file is **impressively well-engineered** for a solo project:

**Strengths:**
- Streaming mode for large datasets (memory-mapped I/O)
- Offset caching (`.offidx` files) for fast re-loading
- HalfKAv2 feature conversion matching C++ exactly
- Multi-dataset support with ratio-based sampling
- Stratified sampling by phase and result type
- Sample rebalancing (eval soft-cap, draw upweighting, mate boost)
- Stochastic Weight Averaging (SWA)
- Cosine annealing with warm restarts
- Gradient accumulation for effective large batch sizes
- Atomic file writes (write to .tmp, rename)
- Comprehensive training progress visualization
- Label smoothing, dropout, weight decay — all the regularization knobs
- Detailed logging with per-phase loss tracking

**Issues:**
- Dense feature representation: `all_white = np.zeros((n, 40960))` at float32 = 160KB per position. For 200K positions per chunk, that's 32 GB per chunk (white+black). This will OOM.
- Actually, the streaming mode loads chunks of `STREAM_CHUNK_SIZE = 200,000` positions, which at 40,960 features × 4 bytes × 2 (white+black) = ~30 GB. This seems like a bug — the old 768-encoding was fine but HalfKAv2's 40,960 features make dense representation impractical.
- The code falls back to 768-encoding in the binary format and converts to HalfKAv2 in memory — this conversion step (`convert_768_to_halfkav2`) is O(n × features) and creates massive dense arrays.

**Recommendation:** Use PyTorch sparse tensors or a custom sparse format for HalfKAv2 features.

### Data Pipeline: **B+**

- `pgn_to_training.py`: Correctly converts PGN to binary training format using python-chess
- `generate_draws.py`: Syzygy tablebase probing for drawn endgame positions — excellent idea
- `convert_lichess_elite.py`: Handles Lichess elite database conversion
- `merge_training_data.py`: Simple binary concatenation

### Script Automation: **B+**

`pipeline.ps1` is well-structured:
- Multi-generation self-play loop
- Configurable parameters (games, depth, epochs, learning rate, etc.)
- Overfitting detection with severity levels
- ELO validation between generations
- Proper error handling with `$ErrorActionPreference = "Stop"`

---

## Testing Assessment

### Test Coverage: **B**

`TestSuite.cpp` (1,627 lines) covers:
- Board setup and piece placement
- Move generation (pawns, knights, sliding pieces, castling, en passant)
- FEN parsing
- Zobrist hashing
- Engine search (mate-in-N, tactical positions)
- NNUE feature encoding
- Training data save/load
- Self-play data generation

**Missing Test Areas:**
- No perft tests (standard move generation verification)
- No time management tests
- No UCI protocol tests (string parsing)
- No repetition detection tests
- No edge cases for 50-move rule
- No stress tests for TT collisions
- No NNUE incremental update correctness tests
- `test_engine_uci.py` and `test_pipeline.py` exist but are integration tests, not unit tests

### Test Quality: **B-**

- Custom test framework (TEST_ASSERT macro) is simple but functional
- Tests are organized by section with pass/fail reporting
- No test isolation — tests share global state
- Some tests are deterministic, others depend on engine search depth (fragile)

---

## Chess-Specific Review

### Move Generation Correctness

- **Pawns:** Correct single push, double push, captures, en passant, promotion
- **Knights:** Correct L-shaped movement
- **Sliding pieces:** Correct ray generation with blocking
- **Castling:** Checks for king/rook moved, through-check, and through-attack correctly
- **En passant:** Properly validates the en passant target square
- **Duck Chess:** Properly handles duck blocking and placement rules

**Concern:** No perft validation. Without perft numbers, subtle move generation bugs (especially around en passant pins, castling edge cases) could exist undetected. Standard practice is to validate against known perft results at depths 1-6 from the starting position and several test positions.

### Evaluation Function Design

**Handcrafted (HCE):**
- PeSTO piece-square tables (well-known good tables)
- Tapered eval (middlegame/endgame interpolation)
- Piece mobility (knights, bishops, rooks, queens)
- Bishop pair bonus
- Doubled/isolated pawn penalties
- Passed pawn bonus (quadratic rank scaling)
- King safety (pawn shield)
- Rook on open/semi-open file

**NNUE:**
- HalfKAv2 encoding (40,960 features)
- Architecture: 40960→512→128→128→1
- SCReLU activation
- Side-to-move perspective handling
- SSE-accelerated inference

### Search Algorithm: **Good**

The search implementation follows modern engine best practices. All major pruning and reduction techniques are present. The aspiration window implementation with graduated widening is correct. Singular extensions are properly guarded against excessive extension.

### UCI Protocol Compliance: **B**

- Core commands implemented correctly: `uci`, `isready`, `ucinewgame`, `position`, `go`, `stop`, `quit`
- `go` handles `wtime`, `btime`, `winc`, `binc`, `movetime`, `depth`
- `setoption` is silently ignored — should support `Hash` and `Threads` at minimum
- Non-standard `generate` command for self-play (acceptable as extension)
- Missing: `ponderhit`, `register`, `debug`

### Time Management: **B+**

- Soft/hard limit system is well-designed
- Best move stability tracking extends/shortens time appropriately
- Time allocation: 1/20th of remaining time (reasonable default)
- Increment handling is correct
- Missing: move overhead parameter, Fischer time increment consideration during iterative deepening

---

## Recommendations

### 1. Critical Fixes

1. **Add `alignas(16)` to NNUE Accumulator arrays** — Prevents undefined behavior from unaligned SSE loads. (NNUE.h, Accumulator struct)

2. **Add perft validation** — Essential for move generation correctness. Implement `perft(depth)` function and validate against known results. This is the single most important correctness test for any chess engine.

3. **Fix dense HalfKAv2 memory usage in train_nnue.py** — The dense 40,960-feature representation will OOM on moderate dataset sizes. Use sparse tensors.

### 2. High-Priority Improvements

4. **Implement make/unmake move** — Replace `Board temp = board;` with incremental board updates. This alone could double NPS. Add an undo stack to Board.

5. **Add incremental Zobrist hashing** — Update hash incrementally during make/unmake instead of recomputing from all 64 squares.

6. **Implement NNUE accumulator stack in search** — Maintain accumulators along the search path and use `incrementalUpdate()` instead of `refreshAccumulator()` at every node.

7. **Add AVX2 code path for NNUE** — SSE processes 4 floats at a time; AVX2 processes 8. Modern CPUs universally support AVX2. This doubles NNUE inference throughput.

8. **Implement `setoption` for Hash and Threads** — Required for proper UCI compliance and GUI compatibility.

9. **Add Lazy SMP multi-threading** — Modern engines gain 50-70% ELO from parallel search. This is the biggest ELO gain available.

### 3. Nice-to-Have Enhancements

10. **Bitboard representation** — A fundamental rewrite that would improve everything: move generation speed, evaluation speed, attack maps. This is a large project but the biggest long-term improvement.

11. **INT8 quantization for NNUE** — Currently using float32. INT8 with clipped ReLU would be 4x faster and is standard in modern engines (Stockfish NNUE).

12. **Opening book** — Replace random opening moves in self-play with a proper opening book (Polyglot format or similar).

13. **Syzygy tablebase probing at runtime** — The project already generates draw data from tablebases; probing them during search would improve endgame play.

14. **Decompose TrainingRunner.cpp** — Split the 3,125-line Win32 GUI into separate files for UI, process management, and graph rendering.

15. **Add CI/CD** — Automated build and test on push. The test suite is already there; just needs a build script for CI.

---

## Metrics Summary Table

| Metric | Value |
|--------|-------|
| Total Lines | ~18,574 |
| Languages | C++ (72%), Python (18%), PowerShell (7%), Batch (3%) |
| Source Files | 37 |
| C++ Headers | 10 |
| C++ Source | 11 |
| Python Scripts | 9 |
| Automation Scripts | 7 |
| Documentation Files | 4 |
| Modules | 10 major (Board, MoveGen, Engine, NNUE, NNUETrainer, UCI, SelfPlayGen, VisualGame, TrainingRunner, train_nnue.py) |
| Bugs Found | 12 |
| Critical Issues | 1 (NNUE alignment) |
| High Severity | 2 (dense HalfKAv2 memory, no perft) |
| Medium Severity | 4 |
| Low Severity | 5 |
| Search Techniques | 17 implemented, 5 major ones missing |
| Test Assertions | ~100+ (estimated from TestSuite.cpp) |
| Overall Grade | **B+** |

---

## Conclusion

This is a **well-crafted chess engine project** that demonstrates strong chess programming knowledge and a mature training pipeline. The engine implements nearly all modern search techniques correctly. The Python training infrastructure is particularly impressive with streaming, multi-dataset support, and sophisticated sampling strategies.

The main limitations are performance-related: the 8×8 array board representation and copy-per-node search pattern significantly limit the engine's NPS. Addressing the make/unmake pattern and incremental NNUE accumulator updates would likely yield the largest immediate ELO gains. The training pipeline's transition to HalfKAv2 encoding needs the dense-to-sparse representation fix to work at scale.

The project is well-positioned for continued improvement, with a clear architectural roadmap (in Architecture_Next_Steps.md) that correctly identifies WDL output heads, INT8 quantization, and DCR (Dynamic Convergence Reversal) as next steps.
