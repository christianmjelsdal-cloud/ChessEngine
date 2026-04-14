# ChessEngine — Project Overview

**Last updated:** April 14, 2026

---

## What It Is

A self-training chess engine with NNUE neural network evaluation, a visual play GUI (SFML), and a Win32 training GUI (TrainingRunner). The engine trains itself through a self-play → train loop, improving each generation.

---

## Architecture

```
ChessEngine.exe          TrainingRunner.exe
├── --uci                ├── Pipeline orchestration
├── --generate           ├── Live graph (loss/val/LR/phase)
├── --generate --duck-chess  ├── Preset management
└── --train-duck         └── Log viewer

Shared core: Board → MoveGen → Engine → NNUE/DuckNNUE
```

---

## Features — Implementation Status

### Search Engine ✅
- Alpha-beta with PVS, LMR, null move, futility, RFP, LMP
- Aspiration windows, killer/history/countermove heuristics, SEE
- Iterative deepening with soft/hard time limits
- Transposition table (4M entries, generation-based replacement)
- Syzygy tablebase probing (5-piece)

### NNUE Evaluation ✅
- Architecture: HalfKAv2 (40,960 features) → L1(512) → L2(128) → L3(64) → 3 phase heads (Opening/Middlegame/Endgame)
- INT16 quantized inference (AVX2/SSE, unaligned loads)
- Incremental accumulator updates (fused copy+update)
- Phase-blended WDL output → centipawn score
- SWA (Stochastic Weight Averaging) in Python trainer

### Board Representation ✅
- 8×8 mailbox + full bitboard redundancy (occupiedBB, colorBB, pieceBBs)
- makeMove/unmakeMove (full snapshot undo — correct but not incremental)
- applyMove (permanent, used by self-play and GUI)
- Zobrist hashing (computed per-position, not incremental)
- FEN import/export

### Self-Play Generation ✅
- Multi-threaded (12 workers, 8MB stack per thread)
- Opening book (FEN file), FRC/Chess960 mix, softmax temperature
- Mixed depth / depth shuffle for throughput
- Adjudication: resign, draw, dead-draw, 50-move, threefold
- Position filtering (min ply, max eval)
- Progress output with nps, ETA, game stats
- `--duck-chess` flag: 832-feature encoding, DuckNNUE weights loaded if available

### Training Pipeline ✅
- **Standard Chess**: `train_nnue.py` (PyTorch, CPU) — full feature set: cosine LR, SWA, WDL loss, label smoothing, grad accumulation, early stopping, phase loss tracking
- **Duck Chess**: `--train-duck` (C++ trainer) — Adam optimizer, train/val split, outputs `loss=`, `val_loss=`, `lr=` per epoch
- Replay window: previous gens' selfplay data mixed in with exponential decay
- Draws dataset mixed in with configurable ratio

### TrainingRunner GUI ✅
- Variant selection: Standard / Duck Chess / Automate Chess (fully separated files)
- 60+ config settings with tooltips, preset save/load
- Live graph: Loss (train+val), Accuracy, LR, Phase Loss — per variant
- Output window: color-coded, overwrite-if-similar for progress lines
- Buttons: Start / Stop / Pause / Skip Phase
- Manual stop vs crash clearly distinguished in logs and output
- Sound effects on self-play complete, gen complete, pipeline complete
- Job Object: child processes killed automatically if TrainingRunner crashes
- Maximized on startup

### Duck Chess Variant ✅
- Full duck chess rules in MoveGen (duck blocks line of sight, no check concept)
- DuckNNUE: 832 features (768 piece + 64 duck square), separate weights
- Self-play generates 832-feature positions
- C++ trainer trains DuckNNUE end-to-end
- Weights feed back into next gen's self-play
- Graph shows train/val loss and LR (no accuracy/phase — C++ trainer limitation)

### Visual Game (ChessEngine.exe GUI) ✅
- SFML 3, piece drag-and-drop, engine vs human/engine
- Live PV arrows, eval bar
- Duck Chess mode toggle (`D` key)
- Automate Chess setup mode (`M` key) — piece placement with budget system

---

## Known Limitations

| Area | Status |
|------|--------|
| makeMove undo | Full board snapshot (correct but ~150 bytes vs ~20 bytes incremental) |
| Duck chess training | No accuracy/phase loss metrics (C++ trainer limitation) |
| Automate Chess | Setup UI done; self-play and training not implemented |
| Lazy SMP | Not implemented — single-threaded search |
| Bitboard move gen | Bitboards maintained but move gen still uses mailbox scan |

---

## File Layout (key files)

```
ChessEngine.exe source:
  Board.cpp/h          — board representation + bitboards
  MoveGen.cpp/h        — legal move generation
  Engine.cpp/h         — search + HCE evaluation
  NNUE.cpp/h           — standard NNUE (40,960 features, quantized)
  DuckNNUE.cpp/h       — duck NNUE (832 features)
  NNUETrainer.cpp/h    — C++ trainer (used by --train-duck)
  SelfPlayGen.cpp/h    — multi-threaded self-play
  UCI.cpp/h            — UCI protocol
  VisualGame*.cpp/h    — SFML GUI
  Main.cpp             — CLI dispatcher

TrainingRunner.exe source:
  TrainingRunner.cpp   — entire Win32 GUI (~3500 lines)

Python training:
  train_nnue.py        — PyTorch trainer for standard NNUE
  assets/openings.txt  — opening book (FEN lines)

Output files (assets/):
  nnue_weights.bin              — current standard weights
  nnue_weights_genN.bin         — per-gen checkpoints
  duck_nnue_weights.bin         — current duck weights
  duck_nnue_weights_genN.bin    — per-gen duck checkpoints
  selfplay_genN.bin             — self-play data
  duck_selfplay_genN.bin        — duck self-play data
  training_data.bin             — base training dataset
  duck_training_data.bin        — base duck training dataset
  logs/training_run_*.log       — structured pipeline logs

Graph persistence:
  training progress/training_graph.csv       — standard chess graph
  training progress/duck_training_graph.csv  — duck chess graph
```
