# Phase 1 — Foundation: Feature Set & Data Pipeline

## Implementation Summary

All four sub-tasks of Phase 1 have been implemented:

---

### 1.1 HalfKAv2 Feature Encoding ✅

**What changed:** The feature space expanded from 768 (12 pieces × 64 squares) to **40,960** features using king-relative encoding: `king_square × 640 + piece_type_index × 64 + piece_square`.

**Key design decisions:**
- **Binary data format unchanged** — Files still store 768-encoding features for backward compatibility. Conversion to HalfKAv2 happens at load time (in Python trainer) and at inference time (in C++ engine).
- **Per-perspective features** — White and Black perspectives are computed independently using their respective king positions. For Black's perspective, all squares are vertically mirrored (`sq ^ 56`).
- **10 piece types per perspective** — 5 own non-king pieces (pawn=0..queen=4) + 5 opponent non-king pieces (pawn=5..queen=9). Kings are not encoded as features; they define the king bucket.

**Files modified:**

| File | Changes |
|------|---------|
| `train_nnue.py` | Added `decode_768_feature()`, `halfkav2_feature()`, `convert_768_to_halfkav2()`. Updated `load_training_data()` and `load_positions_at_offsets()` to convert at load time. `NUM_FEATURES = 40960`. |
| `NNUE.h` | `NUM_FEATURES = 40960`. Added `halfKAv2Feature()`, `squareIndex()`, `mirrorSquare()`, `featureIndex768()`. Accumulator caches king squares. Added `incrementalUpdate()` for efficient move-by-move updates. |
| `NNUE.cpp` | Full HalfKAv2 implementation. `refreshAccumulator()` computes features from board using king buckets. `incrementalUpdate()` handles add/remove/capture; falls back to full refresh on king moves. Weight file version bumped to v2. |
| `NNUETrainer.cpp` | Added `convert768toHalfKAv2()` and `mirrorFeature768()` helpers. Training forward/backward pass converts 768→HalfKAv2 at training time. `extractActiveFeatures()` uses `featureIndex768()`. |

**Incremental accumulator updates:** When a non-king piece moves, only 2-3 features change per perspective (remove from old square, add to new square, optionally remove capture). When the king moves, all features for that perspective change → full refresh. This is the standard HalfKAv2 approach used by Stockfish.

---

### 1.2 Phase Tagging ✅ (was already implemented)

**Verified:** `classify_phase()` in `train_nnue.py` and `analyze_phases.py` correctly tags positions as:
- **Opening** (material phase ≥ 20)
- **Middlegame** (8 ≤ phase < 20)
- **Endgame** (phase < 8)

Material phase weights: Knight=1, Bishop=1, Rook=2, Queen=4.

**New in this phase:** `NNUETrainer.cpp` now computes and stores the phase label in `TrainingPosition.phase` during data generation.

---

### 1.3 Draw Position Tagging ✅

**Draw detection:** Positions with `gameResult ≈ 0.5` are flagged as draws. Draw source types are tracked where determinable:

| Code | Source | Where Detected |
|------|--------|----------------|
| 0 | Not a draw | — |
| 1 | Unknown draw | Self-play adjudication |
| 2 | Stalemate | `pgn_to_training.py` |
| 3 | Insufficient material | `pgn_to_training.py` |
| 4 | 50-move rule | `pgn_to_training.py` |
| 5 | Repetition | `pgn_to_training.py` |
| 6 | Agreement | `pgn_to_training.py` |
| 7 | Tablebase | `generate_draws.py` |

**Files modified:**

| File | Changes |
|------|---------|
| `train_nnue.py` | Added `DrawSource` class and `tag_draw_positions()` function |
| `pgn_to_training.py` | Added `detect_draw_source()` that inspects final board state. Prints draw source distribution at end of run. |
| `generate_draws.py` | Added `DRAW_SOURCE_TABLEBASE = 7` constant. Phase classification for Syzygy positions. |
| `NNUETrainer.h` | `TrainingPosition` gains `drawSource` and `phase` fields |
| `NNUETrainer.cpp` | Draw source tagged during self-play. Binary format v2 includes draw/phase fields (v1 auto-detected and computed). |
| `analyze_phases.py` | New "DRAW STATISTICS" section showing per-phase draw breakdown |

---

### 1.4 Stratified Sampler ✅

**New class: `StratifiedSampler`** in `train_nnue.py`

Categorizes all training positions into 6 pools:
- `decisive-opening`, `decisive-middlegame`, `decisive-endgame`
- `drawn-opening`, `drawn-middlegame`, `drawn-endgame`

Samples from these pools according to configurable ratios, ensuring balanced representation.

**CLI arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--stratified` | off | Enable stratified sampling |
| `--draw-ratio` | 0.15 | Target fraction of draws per batch |
| `--phase-ratio-opening` | 0.33 | Opening sampling weight |
| `--phase-ratio-middlegame` | 0.34 | Middlegame sampling weight |
| `--phase-ratio-endgame` | 0.33 | Endgame sampling weight |
| `--draw-oversample-epochs` | 0 | Oversample draws for first N epochs |
| `--draw-oversample-factor` | 3.0 | Draw oversampling multiplier |

**Example usage:**
```bash
python train_nnue.py \
  --data training_data.bin \
  --stratified \
  --draw-ratio 0.20 \
  --draw-oversample-epochs 10 \
  --draw-oversample-factor 2.5 \
  --phase-ratio-endgame 0.40
```

---

## Binary Format Compatibility

| Format | Version | Fields per Position |
|--------|---------|-------------------|
| v1 (legacy) | 1 | numFeatures, features[], sideToMove, gameResult, searchEval |
| v2 (new) | 2 | v1 + drawSource (uint8), phase (uint8) |

The C++ trainer writes v2 and reads both v1/v2. The Python trainer reads v1 files (ignores missing v2 fields, computes phase/draw at load time). The Python data pipeline scripts (`pgn_to_training.py`, `generate_draws.py`) still write v1 format for maximum compatibility — v2 fields are computed at load time.

---

## Ready for Phase 2

Phase 1 establishes the foundation for Phase 2 (Network Architecture):
- ✅ HalfKAv2 encoding (40,960 sparse features) ready for wider L1 layer
- ✅ Phase and draw tagging ready for multi-head WDL output
- ✅ Stratified sampler ready for balanced training on large datasets
- ✅ Incremental accumulator updates ready for efficient search
