# ChessEngine — Complete Project Overview

**Last updated:** April 17, 2026

---

## Programs

The project produces two executables and one Python script:

| Executable | Purpose |
|---|---|
| `ChessEngine.exe` | Chess engine: UCI mode, self-play generation, duck training, visual GUI |
| `TrainingRunner.exe` | Win32 training pipeline GUI |
| `train_nnue.py` | PyTorch trainer for standard NNUE |

Test projects (`SmokeTests.exe`, `PerftTests.exe`) share the same source tree.

---

## ChessEngine.exe

### CLI Modes (Main.cpp)

Dispatched by command-line flags at startup. Defaults to the SFML visual GUI if no flags are given.

| Flag | Mode | Description |
|---|---|---|
| `--uci` | UCI engine | Standard UCI protocol loop (stdin/stdout) |
| `--generate` | Self-play | Generate training positions to a `.bin` file |
| `--generate --duck-chess` | Duck self-play | Same, with duck chess rules and 832-feature encoding |
| `--generate --automate-chess` | Automate self-play | Automate Chess variant |
| `--train-duck` | Duck training | C++ trainer for DuckNNUE (no Python required) |
| *(no flags)* | Visual GUI | SFML interactive chess GUI |

Startup also performs runtime CPU feature detection (SSE4.1, AVX2, FMA) and prints a clear error if the CPU is unsupported.

---

### Board Representation (Board.cpp/h)

**Dual representation** — mailbox + bitboards kept in sync:

| Field | Type | Purpose |
|---|---|---|
| `squares[8][8]` | `Piece` | Mailbox grid (rank 0 = white's back rank) |
| `occupiedBB` | `Bitboard` | All occupied squares |
| `colorBB[2]` | `Bitboard` | White / Black pieces |
| `pieceBBs[7]` | `Bitboard` | Per piece type (0=None … 6=King) |
| `whiteKingSq` / `blackKingSq` | `Square` | Cached king positions |
| `hash` | `uint64_t` | Incremental Zobrist hash |
| `phase` | `int` | Material phase (0=endgame, 24=opening) |

**Move mechanics:**
- `applyMove(m)` — permanent, no undo. Used by self-play and GUI.
- `makeMove(m, undo)` — reversible. Saves minimal `UndoInfo` snapshot.
- `unmakeMove(m, undo)` — restores from snapshot. Reconstructs `squares[][]` from bitboards (no 512-byte grid snapshot).

**UndoInfo** stores: EP target, castling rights, half-move clock, Zobrist hash, duck square, `isDuckChess` flag, `occupiedBB`, `colorBB[2]`, partial `pieceBBs` snapshot (only the ≤3 changed BBs via `changedMask`/`savedBBs[3]`), king squares, phase.

**Variant flags:**
- `isDuckChess` — enables duck placement rules
- `isAutomateChess` — enables setup-phase budget system (35 points per side, 6 pawns required before pieces)

**FEN support:** `toFEN()` / `fromFEN()` / `recomputeBitboards()`.

---

### Move Generation (MoveGen.cpp/h)

All output uses stack-allocated `MoveList` (no heap allocation).

| Function | Description |
|---|---|
| `getLegalMoves(board, out)` | Full legal move generation |
| `getLegalCaptures(board, out)` | Captures only (for qsearch) |
| `getPseudoLegalMoves(board, out)` | Pseudo-legal (faster, no legality check) |
| `isInCheck(board, color)` | Check detection |
| `getDuckChessMoves(board, out)` | Duck chess: chess moves only |
| `getDuckPlacements(board, out)` | Duck chess: valid duck squares |
| `isKingCaptured(board, color)` | Duck chess win condition |

Piece generators: pawn (including en passant, promotion), knight, bishop, rook, queen, king (including castling). Sliding piece attacks use magic bitboard lookups.

---

### Search Engine (Engine.cpp/h)

**Algorithm:** Iterative deepening alpha-beta with PVS (Principal Variation Search).

#### Pruning & Reductions

| Technique | Condition | Effect |
|---|---|---|
| **Null Move Pruning** | Non-PV, non-check, depth ≥ 3, has non-pawn | R = 3 + depth/6 reduction |
| **Reverse Futility Pruning** | Non-PV, non-check, depth ≤ 3 | Skip if staticEval − 120×depth ≥ beta |
| **Futility Pruning** | Non-PV, non-check, depth ≤ 3 | Skip quiet moves if staticEval + margin ≤ alpha |
| **Late Move Reduction (LMR)** | depth ≥ 3, move index ≥ 3, quiet | reduction = 1 + log(depth)×log(i+1)/2.2 |
| **Late Move Pruning (LMP)** | Non-PV, non-check, depth ≤ 3 | Skip quiets beyond threshold (5/8/13) |
| **SEE Pruning** | Non-PV, depth ≤ 2, capture | Skip if SEE < −100×depth |
| **Check Extension** | In check | depth++ |
| **Internal Iterative Deepening** | PV node, no hash move, depth ≥ 4 | Search at depth−2 first |

LMR adjustments: −1 reduction for killers or high-history moves; continuation history bonus/penalty.

#### Move Ordering

1. Hash move (TT best move)
2. Good captures (MVV-LVA / SEE ≥ 0)
3. Promotions
4. Killer moves (2 per ply)
5. Countermoves
6. History heuristic + 1-ply continuation history
7. Bad captures (SEE < 0)

#### Transposition Table

- **Size:** 4M entries (1 << 22), resizable
- **Entry:** 18 bytes — `uint64_t key`, `int32_t score`, `int16_t depth`, `uint8_t flag`, `uint8_t gen`, `uint16_t best` (packed move)
- **Replacement:** Depth-preferred with generation aging
- **Lazy SMP:** Shared TT pointer for helper threads (`setSharedTT`)

#### Duck Chess Search

Separate `searchDuck()` function:
- Iterates chess moves, then duck placements for each
- Incremental accumulator delta-undo (no 2KB copy per node)
- Null duck pruning: skip duck loop if post-chess eval already beats beta
- **Duck placement TT:** 64K-entry `DuckTTEntry` table keyed by post-chess hash (duck XOR'd out). Caches best duck square for move ordering and full cutoffs.

#### Evaluation

- **Primary:** NNUE (standard or duck variant) via quantized incremental accumulator
- **Fallback:** PeSTO piece-square tables + mobility + pawn structure + king safety
- **Phase blending:** Opening/Middlegame/Endgame heads blended by material phase
- **Accumulator stack:** `accStack_[ply]` for O(1) eval without full refresh
- **Finny Table:** Caches last-refreshed accumulator per king bucket — O(delta) refresh on king moves

#### Lazy SMP

`setThreadCount(N)` spawns N−1 helper threads sharing the main TT. Each helper runs its own iterative deepening loop.

#### Key Constants

| Constant | Value |
|---|---|
| `MATE_SCORE` | 100,000 |
| `MAX_PLY` | 64 |
| `TT_SIZE` | 1 << 22 (4M entries) |
| `SELFPLAY_TT_SIZE` | 1 << 20 (1M entries) |
| `DUCK_TT_SIZE` | 1 << 16 (64K entries) |

---

### NNUE Evaluation (NNUE.cpp/h)

**Architecture:** HalfKAv2

| Layer | Size | Notes |
|---|---|---|
| Input features | 40,960 | 64 king buckets × 10 piece types × 64 squares |
| L1 (per-perspective) | 512 | INT16 quantized accumulator |
| L2 | 128 | INT8 weights, transposed for SIMD |
| L3 | 64 | INT8 weights, transposed for SIMD |
| Output | 1 (scalar) or 3 (WDL) | Phase-blended centipawn score |

**Quantization:**
- L1: INT16, QA = 256
- L2/L3 activations: uint8, QA_ACT = 127
- L2/L3 weights: INT8, QW = 64

**Inference path:**
1. Refresh or incrementally update `QAccumulator` (INT16, 512 values per perspective)
2. SCReLU activation: clamp(x, 0, 1)²
3. INT8 GEMV for L2 and L3 (AVX2 `_mm256_maddubs_epi16`)
4. Output dot product → centipawn score

**Finny Table:** 128 entries (2 perspectives × 64 king squares). On king move, only the delta from the cached entry is applied — avoids full 40,960-feature rebuild.

**Memory:** ~85MB per instance (L1 weights heap-allocated via `unique_ptr` to avoid MSVC compiler heap limit C1060).

**Legacy encoding:** 768-feature index (`featureIndex768`) for data file compatibility.

---

### DuckNNUE (DuckNNUE.cpp/h)

Same architecture as standard NNUE with one difference:

| | Standard NNUE | DuckNNUE |
|---|---|---|
| Features | 40,960 (HalfKAv2) | 832 (768 piece + 64 duck square) |
| L1/L2/L3 | 512/128/64 | 512/128/64 (same) |
| Quantization | Same | Same |
| Mirror | `mirrorFeature(i) = i ^ 384` | `mirrorDuckFeature(i)` — flips rank for duck features |

Duck feature index: `768 + rank*8 + col`. The duck square is encoded as a one-hot feature appended to the standard 768-feature vector.

---

### Self-Play Generation (SelfPlayGen.cpp/h)

**Binary format** (compatible with `train_nnue.py`):
```
Header:  "NNUE" magic | uint8 version=1 | uint32 position_count
Records: uint16 num_features | uint16 feature_indices[] | uint8 stm | float32 result | float32 eval
```

**Worker model:** Each of N workers owns its own `Engine` instance (heap-allocated, 8MB stack via `CreateThread`). Workers write to pre-allocated per-game storage (no locking on position writes). A separate countdown thread ticks every second to update the ETA display.

**Game diversity features:**

| Feature | Parameter | Default |
|---|---|---|
| Opening book | `--openings` | FEN file, one per line |
| Softmax opening | `openingTemp` | 1.5 (4 plies) |
| Softmax post-opening | `softmaxTemp` | 0.5 (8 plies) |
| Root noise | `rootNoiseEps` | 0.0 (off) |
| Chess960 mix | `frcMix` | 0.0 |
| Mixed depth | `mixedDepthRatio` | 0.0 |
| Depth shuffle | `depthShuffle` | false |

**Adjudication:**

| Rule | Threshold | Count | Min ply |
|---|---|---|---|
| Resign | 500 cp | 3 consecutive | — |
| Draw | 8 cp | 6 consecutive | 40 |
| Dead draw | 4 cp | 12 consecutive | 50 |
| 50-move | halfMoveClock ≥ 100 | — | — |
| Max plies | 250 | — | — |
| Timeout | 120s per game | — | — |

**Position filtering:** Skip positions before ply 10 (`recordMinPly`) or with |eval| > 2500 cp (`recordMaxEval`).

**NPS sampling:** Emits `NPS_SAMPLE step=N nps=X` lines at evenly-spaced game intervals (one per epoch slot) for the training graph.

---

### UCI Protocol (UCI.cpp/h)

Supports the full standard UCI command set:

| Command | Notes |
|---|---|
| `uci` | Engine identification |
| `isready` | Readiness ping |
| `ucinewgame` | Reset state |
| `position [fen/startpos] moves ...` | Set board |
| `go [depth/movetime/wtime/btime/...]` | Start search |
| `stop` | Halt search |
| `ponderhit` | Resume from ponder |
| `quit` | Exit |

Non-standard: `generate` command for self-play generation from UCI context.

Pondering is supported via atomic flags (`ponderEnabled_`, `isPondering_`). Search runs on a dedicated thread; `coutMutex_` serializes output.

---

### Visual GUI (VisualGame.cpp/h, SFML 3)

**Board interaction:**
- Click-to-select or drag-and-drop piece movement
- Promotion dialog (piece selection)
- Smooth move animation (0.15s)
- Legal move highlighting

**Game history & analysis:**
- Full `HistoryEntry` vector (board snapshot, move, algebraic notation, eval, centipawn loss)
- Navigate with ← / → arrow keys or scroll wheel
- Analysis mode: dedicated `analysisEngine_` runs on the viewed position
- Debounced analysis start (250ms after navigation stops)

**Analysis panel (right side):**
- Top 3 PV lines with eval pill, move continuation, expand/collapse toggle
- Live stats: depth, nodes, NPS, elapsed
- Eval bar (white/black advantage)
- Centipawn loss per move (color-coded: Best/Good/Inaccuracy/Mistake/Blunder)
- Opening name (ECO database, 3,695 openings)
- Syzygy tablebase result for ≤5-piece endgames

**Engine modes:**
- Human vs Engine
- Bot vs Bot (two separate `Engine` instances)
- Engine color and time configurable

**Algebraic notation:** Full SAN with disambiguation, check/checkmate suffix, castling, en passant.

**Variant support:** Duck Chess (`D` key toggle), Automate Chess (`M` key for setup mode).

---

### Game Logic (GameLogic.cpp/h)

Pure, reusable rule functions with no GUI dependency.

**Result classification** (`classify()`):

| Result | Condition |
|---|---|
| `CheckmateWhiteWins` / `CheckmateBlackWins` | Side to move has no legal moves and is in check |
| `Stalemate` | No legal moves, not in check |
| `DrawByRepetition` | Threefold repetition (Zobrist hash) |
| `DrawBy50Move` | halfMoveClock ≥ 100 |
| `DrawByInsufficientMaterial` | K vs K, K+B vs K, K+N vs K, K+B vs K+B (same color) |
| `DuckNoMovesWhiteWins` / `DuckNoMovesBlackWins` | Duck chess: no legal moves (no stalemate concept) |
| `DuckKingCapturedWhiteWins` / `DuckKingCapturedBlackWins` | Duck chess: king captured |

---

### Syzygy Tablebases (Syzygy.cpp/h)

- **WDL probe:** Returns score from side-to-move perspective. Used in search at leaf nodes.
- **Root probe (DTZ):** Returns best move + WDL score. Used at root for optimal endgame play.
- **Auto-detection:** Loads from `Syzygy345/` directory adjacent to the executable.
- **Score constants:** `TB_WIN_SCORE` = 20,000 (< MATE, >> eval), `TB_CURSED_WIN` = 50, `TB_BLESSED_LOSS` = −50.
- **Condition:** Probe only when piece count ≤ `maxPieces()` and no castling rights.

---

## TrainingRunner.exe

Standalone Win32 training pipeline GUI. No browser, no server, no network required.

### Architecture

- **Win32 + GDI+** for rendering (no SFML dependency)
- **Pipeline thread** runs self-play and training as child processes via `RunProc()`
- **Job Object** ensures child processes are killed if TrainingRunner crashes
- **WM_TIMER** (500ms) drives live graph updates, ETA countdowns, and log flushing

### Variants

Three training variants selectable from a dropdown:

| Variant | Self-play flag | Training | Weights prefix |
|---|---|---|---|
| Standard Chess | `--generate` | `train_nnue.py` | `nnue_weights` |
| Duck Chess | `--generate --duck-chess` | `--train-duck` | `duck_nnue_weights` |
| Automate Chess | `--generate --automate-chess` | *(not yet implemented)* | `automate_play_weights` |

### Pipeline Phases

Each generation runs these phases in order:

1. **Self-play** — `ChessEngine.exe --generate ...` — produces `selfplay_genN.bin`
2. **Training** — `train_nnue.py` (standard) or `ChessEngine.exe --train-duck` (duck) — produces `nnue_weights.bin`
3. **ELO Validation** *(optional)* — `cutechess-cli` match vs previous gen
4. **SWA** *(optional)* — Stochastic Weight Averaging match

### Configuration (60+ parameters)

Organized into groups:

**Pipeline:**
- Generations, start gen, variant

**Self-play:**
- Games/gen, workers, depth, mixed depth ratio/low, depth shuffle/bias
- Resign CP/count, contempt, max plies, draw CP
- Opening temp/plies, softmax plies/temp, root noise
- Record min ply, record max eval
- Draw adj moves/threshold/min move, draw count, draw min ply
- FRC mix

**Training:**
- Epochs/gen, batch size, learning rate, weight decay, dropout, label smoothing
- Gradient accumulation, warmup steps, draw weight, mate boost
- Self-play ratio, replay window/decay, max positions, early stop
- Cosine LR (with optional restart period T0)
- SWA (start epoch)
- Draw dataset %, WDL alpha, WDL draw ELO

**Validation:**
- ELO validation toggle, ELO games, SWA games

### Preset System

- 4 built-in presets: Quick Test, 1 Hour, Standard, Aggressive
- Unlimited custom presets (saved to `custom_presets.cfg`)
- Save / Delete / Load from dropdown

### Live Graph (GDI+, double-buffered)

Five panels, each toggleable:

| Panel | Content |
|---|---|
| **Loss** | Train (blue) + Val (orange) curves, best markers, gen boundaries |
| **Accuracy** | Move-prediction accuracy on held-out positions |
| **Learning Rate** | LR schedule (cosine decay, restarts) |
| **Phase Loss** | Opening / Middlegame / Endgame loss curves |
| **NPS (Self-Play)** | Nodes per second during self-play, aligned to epoch x-positions |

Graph features:
- Hover crosshair + tooltip (step, gen, train, val, LR, acc, phase losses, NPS)
- Generation boundary lines with labels
- Best-value diamond markers
- PNG export (`SaveGraphPng`)
- Persistent CSV (`training progress/[variant_]training_graph.csv`) — survives restarts
- Dirty-flag cache: only re-copies `g_st.pts` when data changes

### Output Window

- Color-coded log lines (self-play = teal, training = orange, success = green, error = red)
- `\r`-prefixed lines overwrite the previous line (running progress updates)
- Non-`\r` lines append as new entries
- Max 800 lines (circular buffer)

### Banner & ETA

Live banner updates every 500ms:
```
Phase: Self-play  Gen: 1/10  Games: 449/500  |  Phase elapsed: 10m 24s  |  Self-play done in: ~46s  |  Pipeline elapsed: 10m 24s
```

Training banner:
```
Phase: Training  Gen: 1/10  Epoch: 3/20  Batch: 127/556  |  Phase elapsed: 2m 10s  |  Batch ETA: ~8s  |  Training done in: ~2m 39s
```

Countdown fields: `selfPlayEtaSec`, `batchEtaSec`, `epochEtaSec`, `nextEpochSec` — each with a timestamp for live countdown.

### Controls

- **Start / Stop / Pause / Skip Phase** buttons
- **Latest Gen** / **Best Gen** buttons (load weights from best val-loss gen)
- **Benchmark** button (measures games/sec at current depth)
- **Mute Sounds** checkbox
- Sound effects: self-play complete, gen complete, pipeline complete (MP3 via MCI)

### File Logging

Structured log to `assets/logs/training_run_YYYYMMDD_HHMMSS.log`:
```
[2026-04-17T02:40:12.345] [INFO] [selfplay:4] [SelfPlay] 449/500 (89%) ...
[2026-04-17T02:41:05.123] [METRIC] [training:4] EPOCH epoch=3 train_loss=0.02707 val_loss=0.00990 lr=0.00010000
```

---

## NNUETrainer (NNUETrainer.cpp/h)

C++ trainer used for duck chess (`--train-duck`). Also contains the self-play data generator used by the standard training path.

### Training Loop (trainDuck)

**Optimizer:** AdamW (Adam + decoupled weight decay)
- β1 = 0.9, β2 = 0.999, ε = 1e-8
- LR warmup: linear ramp over `warmupSteps` batches
- Cosine LR annealing with optional warm restarts (T0)
- Gradient accumulation (effective batch = batchSize × gradAccum)
- SWA: running weight average from `swaStart` epoch onward

**Loss function:**
```
loss = λ × (sigmoid(pred/scale) − sigmoid(eval/scale))²
     + (1−λ) × (sigmoid(pred/scale) − gameResult)²
```
With optional mate boost (upweight positions with |eval| > 300 cp).

**Parallelism:** Persistent thread pool (created once, reused for all batches):
- `numTrainThreads` = `hardware_concurrency()`, capped at 16
- Workers sleep on a condition variable between batches
- Zero thread creation overhead across all batches in all epochs

**AVX2 forward/backward pass** (`TrainAVX.h`):

| Operation | AVX2 primitive | Speedup |
|---|---|---|
| L1 accumulation | `avx_add_row<512>` | ~8× vs scalar |
| SCReLU activation | `avx_screlu<N>` | ~8× |
| L2/L3 GEMV | `avx_gemv_T<OUT,IN>` | ~6-8× |
| Output dot product | `avx_dot<64>` | ~8× |
| Weight gradient (outer product) | `avx_outer_add<ROWS,COLS>` | ~8× |
| Backward L1/L2 propagation | `avx_matvec_T_add<ROWS,COLS>` | ~6-8× |

All primitives fall back to scalar if `__AVX2__` is not defined.

**Phase-balanced training:** Positions classified as Opening/Middlegame/Endgame by material count. Oversamples minority phases to balance the training distribution.

**Data sources:** Multiple `--extra-data file ratio` arguments. Positions capped at `maxPositions` (random shuffle before cap).

---

## train_nnue.py

PyTorch-based trainer for the standard NNUE (HalfKAv2, 40,960 features).

### Features

- **Streaming data loading:** Memory-mapped temp files for datasets > 256MB — no RAM limit
- **Multi-dataset mixing:** `--extra-data file ratio` for self-play + base data blending
- **Replay window:** Previous gens' self-play data mixed with exponential decay
- **Loss:** Same sigmoid MSE + WDL blend as C++ trainer
- **LR schedule:** Cosine annealing with optional warm restarts, minimum LR floor
- **SWA:** Stochastic Weight Averaging from configurable start epoch
- **Early stopping:** Patience-based on validation loss
- **Phase loss tracking:** Per-epoch Opening/Middlegame/Endgame loss breakdown
- **Validation split:** 90/10 train/val
- **Data validation:** Sanity checks on `game_result` and `stm` values
- **Enhanced mode (`--enhanced`):** Full feature set (cosine LR, SWA, WDL loss, label smoothing, grad accumulation, phase loss)

### Key flags

```
--data FILE           Base training data
--extra-data FILE W   Additional data with weight W
--epochs N            Training epochs
--batch-size N        Mini-batch size
--lr F                Initial learning rate
--lr-min F            Minimum LR floor
--weight-decay F      AdamW weight decay
--dropout F           Dropout rate
--grad-clip F         Gradient clipping
--grad-accum N        Gradient accumulation steps
--warmup-steps N      LR warmup steps
--draw-weight F       Draw position upweight
--mate-boost F        Near-mate position upweight
--early-stop N        Early stopping patience
--wdl-alpha F         WDL blend (0=MSE only, 1=CE only)
--wdl-draw-elo F      Draw bandwidth in centipawns
--cosine-lr           Enable cosine LR
--cosine-t0 N         Cosine restart period
--swa                 Enable SWA
--swa-start N         SWA start epoch
--label-smoothing F   Label smoothing on game result
--load-weights FILE   Resume from checkpoint
--output FILE         Output weights path
--enhanced            Enable all advanced features
--plot                Save training graph PNG
```

---

## Shared Infrastructure

### Bitboard.cpp/h

Magic bitboard attack tables for sliding pieces (bishop, rook, queen). Used by `MoveGen` for fast attack generation and by `isSquareAttacked`.

### Types.h

Core types:
- `PieceType` enum: None, Pawn, Knight, Bishop, Rook, Queen, King, Duck (DUCK_CHESS only)
- `Color` enum: White, Black
- `Square` struct: `int rank, col` (−1 = invalid)
- `Piece` struct: `PieceType type, Color color`
- `Move` struct: `Square from, to`, `PieceType promotion`, `Square duckTo` (DUCK_CHESS only)
- `MoveList`: stack-allocated array of up to 256 moves
- `SquareList`: stack-allocated array of up to 64 squares

### TrainAVX.h

AVX2 training primitives (header-only, auto-detects `__AVX2__`):
- `avx_add_row<N>` — accumulator update
- `avx_axpy<N>` — fused multiply-add
- `avx_screlu<N>` — SCReLU with pre-activation save
- `avx_screlu_deriv_mul<N>` — SCReLU backward
- `avx_gemv_T<OUT,IN>` — GEMV with transposed matrix
- `avx_dot<N>` — dot product
- `avx_outer_add<ROWS,COLS>` — outer product accumulate
- `avx_matvec_T_add<ROWS,COLS>` — backward matrix-vector

---

## Key Sizes & Constants

| Item | Value |
|---|---|
| Standard NNUE features | 40,960 (HalfKAv2) |
| DuckNNUE features | 832 (768 + 64 duck) |
| L1 neurons (per perspective) | 512 |
| L2 neurons | 128 |
| L3 neurons | 64 |
| NNUE memory | ~85 MB |
| TT size (UCI/GUI) | 4M entries (1 << 22) |
| TT size (self-play) | 1M entries (1 << 20) |
| Duck TT size | 64K entries (1 << 16) |
| TT entry size | 18 bytes |
| MAX_PLY | 64 |
| MATE_SCORE | 100,000 |
| TB_WIN_SCORE | 20,000 |
| L1 quantization (QA) | 256 |
| L2/L3 activation (QA_ACT) | 127 |
| L2/L3 weight scale (QW) | 64 |
| Finny table entries | 128 (2 × 64) |
| Continuation history | 6×64×6×64 = 147,456 entries |
| Max log lines (output window) | 800 |
| Max training positions (default) | 300,000 |

---

## Variant Support Summary

| Feature | Standard | Duck Chess | Automate Chess |
|---|---|---|---|
| Network | NNUE (40,960 feat.) | DuckNNUE (832 feat.) | AutomateNNUE (TBD) |
| Self-play flag | `--generate` | `--generate --duck-chess` | `--generate --automate-chess` |
| Training | `train_nnue.py` | `--train-duck` (C++) | Not implemented |
| Win condition | Checkmate | King captured | Checkmate |
| Stalemate | Draw | No moves = loss | Draw |
| Setup phase | No | No | Yes (35-point budget) |
| FRC/Chess960 | Yes (`--frc-mix`) | No | No |
| Syzygy | Yes | No | No |
| Weights prefix | `nnue_weights` | `duck_nnue_weights` | `automate_play_weights` |
| Graph CSV | `training_graph.csv` | `duck_training_graph.csv` | `automate_training_graph.csv` |
