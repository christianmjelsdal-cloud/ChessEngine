# Architecture Next Steps — Gap Analysis Report
**Generated:** 15 March 2026  
**Scope:** Compare `Architecture_Next_Steps.md` phases/steps against current codebase state

---

## Executive Summary

| Phase | Steps | ✅ Done | ⚠️ Partial | ❌ Not Started |
|-------|-------|---------|------------|----------------|
| **1 — Foundation** | 4 | 4 | 0 | 0 |
| **2 — Network Architecture** | 4 | 3 | 1 | 0 |
| **3 — Training Script Updates** | 4 | 2 | 1 | 1 |
| **4 — Dynamic Convergence Reversal** | 5 | 0 | 0 | 5 |
| **5 — Quantization & Inference** | 3 | 1 | 1 | 1 |
| **Totals** | **20** | **10** | **3** | **7** |

**Overall completion: 50% done, 15% partial, 35% not started.**

Phase 1 (Foundation) and Phase 2 (Network) are essentially complete. Phase 3 (Training) is mostly done but missing WDL label generation. Phase 4 (DCR) is entirely unimplemented. Phase 5 (Quantization) is partially done but uses INT16 instead of INT8.

---

## Phase 1 — Foundation: Feature Set & Data Pipeline

### 1.1 Adopt HalfKAv2 Feature Encoding — ✅ COMPLETE

| Planned | Implemented |
|---------|------------|
| 64 king buckets × 10 non-king types × 64 squares = 40,960 features | `NNUE.h`: `NUM_FEATURES = 40960`, `KING_BUCKETS = 64`, `PIECES_PER_PERSPECTIVE = 10` |
| Per-perspective feature computation | `halfKAv2Feature()` in C++; `halfkav2_feature()` in Python |
| Mirror squares for black perspective | `mirrorSquare()` in C++; `king_sq ^ 56` in Python |
| Backward-compatible with 768-encoding data files | Binary files store 768-encoding; `convert_768_to_halfkav2()` converts at load/train time |

**Key files:** `NNUE.h`, `NNUE.cpp`, `train_nnue.py`

**Notes:** Clean separation between storage format (768) and runtime format (HalfKAv2). Both C++ and Python implementations are in sync.

---

### 1.2 Tag Positions by Game Phase — ✅ COMPLETE

| Planned | Implemented |
|---------|------------|
| Material-based phase scoring | `computeMaterialPhase()` in C++; `compute_material_phase()` in Python |
| Three-tier classification (opening/mid/end) | C++: thresholds ≥20/≥8/<8. Python: identical thresholds |
| Phase weights: Knight/Bishop=1, Rook=2, Queen=4 | Consistent across all files |
| Phase field in training data | `TrainingPosition::phase` (uint8_t) in `NNUETrainer.h` |

**Key files:** `NNUETrainer.h`, `NNUETrainer.cpp`, `train_nnue.py`, `analyze_phases.py`, `generate_draws.py`

**Notes:** Phase classification is implemented identically in 5 separate files. `analyze_phases.py` provides a standalone analysis tool for dataset phase distribution.

---

### 1.3 Tag Drawn Positions — ✅ COMPLETE

| Planned | Implemented |
|---------|------------|
| Draw source enum | `DrawSource` enum in `train_nnue.py` (NOT_DRAW through TABLEBASE, 0–7) |
| Tag positions from game results | `tag_draw_positions()` function |
| Draw source field in training data | `TrainingPosition::drawSource` (uint8_t) in `NNUETrainer.h` |
| Syzygy tablebase draw tagging | `generate_draws.py` uses `DRAW_SOURCE_TABLEBASE = 7` |

**Key files:** `train_nnue.py`, `NNUETrainer.h`, `NNUETrainer.cpp`, `generate_draws.py`

**Notes:** Self-play data tags draws as `UNKNOWN_DRAW` (1). Syzygy data explicitly tagged as `TABLEBASE` (7). The draw source is NOT embedded in the binary format — it's tracked by file origin.

---

### 1.4 Build a Stratified Sampler — ✅ COMPLETE

| Planned | Implemented |
|---------|------------|
| Pool positions by (is_draw, phase) | `StratifiedSampler` class with 6 pools: decisive/drawn × 3 phases |
| Configurable draw ratio | `draw_ratio` parameter (default 0.15) |
| Configurable phase ratios | `phase_ratios` dict `{0: 0.33, 1: 0.34, 2: 0.33}` |
| Draw oversampling schedule | `draw_oversample_epochs` and `draw_oversample_factor` |
| CLI activation | `--stratified` flag with supporting args |

**Key files:** `train_nnue.py` (class `StratifiedSampler`)

**Notes:** Activated via `--stratified` flag. Uses replacement sampling for underrepresented pools. Automatically pads/trims to exact epoch size.

---

## Phase 2 — Network Architecture

### 2.1 Widen L1 to 1024 — ✅ COMPLETE (Exceeds Plan)

| Planned | Implemented | Delta |
|---------|------------|-------|
| L1: 1024 | `L1_SIZE = 1024` | ✅ Match |
| L2: 64 | `L2_SIZE = 128` | ↑ 2× larger |
| L3: 32 | `L3_SIZE = 64` | ↑ 2× larger |
| Activation: Clipped ReLU | `SCReLU` (squared clipped ReLU) | ↑ Better activation |

**Key files:** `NNUE.h`, `train_nnue.py`

**Notes:** Network is larger than planned — L2 and L3 are each 2× wider. SCReLU (`clamp(x, 0, 1)²`) replaces clipped ReLU, which is generally better for NNUE training. These are improvements over the plan, not gaps.

---

### 2.2 Add WDL Output Head — ✅ COMPLETE

| Planned | Implemented |
|---------|------------|
| 3-output WDL (win/draw/loss) | `WDL_SIZE = 3` in NNUE.h; `PhaseHead` struct with `weights[3][L3_SIZE]` |
| Softmax over WDL logits | `forward()` in NNUE.cpp computes softmax |
| Score = (P(win) - P(loss)) × 400 | `(wdl[0] - wdl[2]) * 400` in C++ `forward()` |

**Key files:** `NNUE.h`, `NNUE.cpp`, `train_nnue.py`

---

### 2.3 Implement Phase-Blended Multi-Head Output — ✅ COMPLETE

| Planned | Implemented |
|---------|------------|
| Three separate phase heads | `head_opening`, `head_middlegame`, `head_endgame` (both C++ and Python) |
| Quadratic blending weights | C++: `w_op = p²`, `w_eg = (1-p)²`, `w_mg = 2p(1-p)` |
| Material-based runtime phase | `computePhase()` returns float in [0,1] |

**Key files:** `NNUE.h`, `NNUE.cpp`, `train_nnue.py`

**Notes:** C++ uses continuous quadratic blending for inference. Python training uses discrete per-position head selection (`torch.where(pm == 0, wdl_o, ...)`) which is a simpler approach that avoids backprop complexity of blending during training.

---

### 2.4 Network Summary — ⚠️ PARTIAL (Improved Beyond Plan)

| Planned Architecture | Actual Architecture |
|---------------------|-------------------|
| Input(40960) → L1(1024, ClipReLU) → L2(64, ClipReLU) → L3(32, ClipReLU) | Input(40960) → L1(1024, SCReLU) → L2(128, SCReLU) → L3(64, SCReLU) |
| → 3 heads × WDL(3) | → 3 heads × WDL(3) |
| ~42M parameters | **~84M parameters** (L1 dominates, but L2/L3 are 4× larger in total) |

**Status:** Marked partial only because it deviates from the spec, but all deviations are **improvements** (wider layers, better activation). If the doc is treated as a minimum spec, this is complete. If treated as exact spec, the L2/L3 sizes differ.

---

## Phase 3 — Training Script Updates

### 3.1 Per-Head Loss Tracking — ✅ COMPLETE

| Planned | Implemented |
|---------|------------|
| Track loss per phase head | Validation loss computed per-phase via `val_phase_indices` |
| Log per-head loss | CSV columns: `opening_loss`, `middlegame_loss`, `endgame_loss` |
| Display per-head loss | Console output: `Phase loss -> Opening: X.XX Middlegame: X.XX Endgame: X.XX` |
| Plot per-head loss | Separate "Phase Loss" subplot in `training_progress.png` |

**Key files:** `train_nnue.py` (training loop, `append_log()`, `generate_plot()`)

---

### 3.2 Per-Phase Loss Weighting — ⚠️ PARTIAL

| Planned | Implemented | Gap |
|---------|------------|-----|
| Static per-phase weights | CLI args: `--weight-opening`, `--weight-middlegame`, `--weight-endgame` | ✅ Done |
| Dynamic adjustment from DCR | Not implemented | ❌ Requires Phase 4 |

**Key files:** `train_nnue.py` (argparse section, training loop)

**Notes:** Static weights work and are applied as multipliers. Dynamic adjustment depends on Phase 4 (DCR) which is entirely unimplemented.

---

### 3.3 WDL Label Generation — ❌ NOT IMPLEMENTED

| Planned | Current State | Gap |
|---------|--------------|-----|
| Convert game result + eval → WDL ground truth `[P(win), P(draw), P(loss)]` | Not implemented | ❌ |
| Sigmoid-based WDL estimate blended with game result | Not implemented | ❌ |
| Cross-entropy loss against WDL labels | Not implemented | ❌ |
| Configurable blend factor | Not implemented | ❌ |

**Current approach:** The training loss uses **sigmoid MSE** between a scalar eval proxy `(win_logit - loss_logit)` and the target `(λ * sigmoid(eval/scale) + (1-λ) * game_result)`. This indirectly trains the WDL heads through the scalar proxy, but does NOT directly optimize the WDL distribution with cross-entropy.

**Impact:** The WDL heads are only trained through the gradient of the scalar difference `(win - loss)`, meaning the **draw logit receives no direct training signal**. This is a significant gap — the network learns to distinguish "white winning" from "black winning" but has no explicit incentive to produce calibrated draw probabilities.

**Recommended fix:**
1. Add WDL ground truth generation: `P(win) = result × (1-blend) + sigmoid(eval/scale) × blend`, `P(loss) = (1-result) × (1-blend) + sigmoid(-eval/scale) × blend`, `P(draw) = 1 - P(win) - P(loss)`
2. Add cross-entropy loss: `-Σ [y_wdl × log(softmax(wdl_logits))]`
3. Blend with existing eval loss: `total = α * CE_loss + (1-α) * eval_MSE_loss`

---

### 3.4 Drawn Position Oversampling Schedule — ✅ COMPLETE

| Planned | Implemented |
|---------|------------|
| Configurable oversample factor and epoch count | `draw_oversample_factor` (default 3.0), `draw_oversample_epochs` (default 0) |
| Integrated into stratified sampler | `StratifiedSampler.sample_epoch_indices()` applies oversampling for first N epochs |
| CLI args | `--draw-oversample-epochs`, `--draw-oversample-factor` |
| C++ trainer support | `TrainingConfig::drawOversampleFactor`, `drawOversampleEpochs` |

**Key files:** `train_nnue.py`, `NNUETrainer.h`

---

## Phase 4 — Dynamic Convergence Reversal (DCR)

### ❌ ENTIRELY NOT IMPLEMENTED (0/5 Steps)

| Step | Planned | Status |
|------|---------|--------|
| **4.1** Divergence Detection | Rolling window monitoring of per-phase loss, compute divergence metric | ❌ |
| **4.2** Continuous Reweighting | Dynamic per-phase loss weight adjustment based on divergence | ❌ |
| **4.3** Data Composition Blending | Adjust draw/decisive ratio and phase sampling in real-time | ❌ |
| **4.4** GUI Integration | DCR toggle, divergence display, intervention logging | ❌ |
| **4.5** Safeguards | Max weight bounds, cooldown periods, manual override | ❌ |

**Dependencies satisfied:** Per-head loss tracking (3.1) ✅, Per-phase loss weighting (3.2 static) ✅, Stratified sampler (1.4) ✅

**Notes:** All prerequisite infrastructure (per-phase loss, static weights, stratified sampling) is in place. DCR requires building the dynamic feedback loop on top of these foundations. This is the largest remaining gap in the architecture plan.

**Estimated effort:** Medium-large. Core algorithm (~200 LOC in Python), GUI integration is optional.

---

## Phase 5 — Quantization & Inference

### 5.1 INT8 Quantization — ⚠️ PARTIAL (INT16 Implemented, Not INT8)

| Planned | Implemented | Gap |
|---------|------------|-----|
| INT8 weights and activations | **INT16** quantization with QA=256 | Different bit width |
| Quantize L1 feature transform | `L1_weights_q` (int16_t), `L1_biases_q` (int16_t) | ✅ (but INT16) |
| Quantized forward pass | `forwardQ()`, `evaluateQ()`, `refreshAccumulatorQ()` | ✅ (but INT16) |
| Saturating arithmetic | Saturating add/sub with INT16_MAX/MIN clamping (our patch) | ✅ |
| QAT (quantization-aware training) | Not implemented | ❌ |

**Key files:** `NNUE.h`, `NNUE.cpp`

**Notes:** INT16 quantization is fully functional and may actually be the better choice for a CPU-only engine (INT8 requires careful range management and SIMD support). However, if the plan specifically requires INT8, the quantization scale, types, and forward pass all need updating. The doc mentions "INT8" explicitly.

**Current quantization path:**
- `quantizeWeights()` → converts float L1 weights to INT16 with scale QA=256
- `refreshAccumulatorQ()` → INT16 accumulator refresh
- `incrementalUpdateQ()` → INT16 incremental updates
- `forwardQ()` → INT16/float hybrid forward (L1 is INT16, L2/L3/heads remain float)

---

### 5.2 Tapered Blend at Runtime — ✅ COMPLETE

| Planned | Implemented |
|---------|------------|
| Material-based phase → blend weights | `computePhase()` → float in [0,1] |
| Quadratic blending of phase heads | `w_op = p²`, `w_eg = (1-p)²`, `w_mg = 2p(1-p)` |
| Applied to both float and quantized paths | `forward()` and `forwardQ()` both use blending |

**Key files:** `NNUE.cpp`

---

### 5.3 Benchmark Suite — ❌ NOT IMPLEMENTED

| Planned | Current State | Gap |
|---------|--------------|-----|
| Formal nodes/second benchmark | Not implemented | ❌ |
| A/B strength testing framework | `estimateElo()` exists but is not a formal benchmark | ⚠️ |
| Quantized vs float comparison | No automated comparison | ❌ |

**Notes:** `NNUETrainer::estimateElo()` provides basic ELO estimation by playing the NNUE engine against the handcrafted engine, but there's no formal benchmark infrastructure for measuring inference speed, comparing quantized vs float accuracy, or running standardized test suites (e.g., WAC, BT2450, STS).

---

## Summary of Remaining Work

### High Priority (blocks further progress)
| Item | Phase | Effort | Impact |
|------|-------|--------|--------|
| WDL Label Generation + Cross-Entropy Loss | 3.3 | Medium | Critical — draw head receives no direct training signal |

### Medium Priority (major planned features)
| Item | Phase | Effort | Impact |
|------|-------|--------|--------|
| DCR: Divergence Detection | 4.1 | Medium | Core of the DCR system |
| DCR: Continuous Reweighting | 4.2 | Medium | Automates phase balancing |
| DCR: Data Composition Blending | 4.3 | Medium | Dynamic draw/decisive ratio |
| DCR: Safeguards | 4.5 | Small | Prevents runaway weights |

### Lower Priority (polish and optimization)
| Item | Phase | Effort | Impact |
|------|-------|--------|--------|
| INT8 Quantization (if INT16 insufficient) | 5.1 | Large | ~2× inference speedup vs INT16 |
| QAT (quantization-aware training) | 5.1 | Medium | Better quantized accuracy |
| Benchmark Suite | 5.3 | Medium | Automated regression testing |
| DCR: GUI Integration | 4.4 | Medium | Nice-to-have visualization |

### Deviations That Are Improvements (No Action Needed)
| Item | Plan | Actual | Assessment |
|------|------|--------|-----------|
| L2 width | 64 | 128 | More capacity — keep |
| L3 width | 32 | 64 | More capacity — keep |
| Activation | Clipped ReLU | SCReLU | Better gradient flow — keep |
| Quantization | INT8 | INT16 | Wider range, simpler — evaluate if INT8 needed |

---

## Recommended Next Step

**Implement Step 3.3 (WDL Label Generation)** first. This is the single highest-impact gap because:
1. The WDL heads already exist in the network but the draw logit has no direct training signal
2. It's a prerequisite for meaningful DCR (Phase 4) — you can't dynamically balance WDL convergence if the W/D/L targets aren't properly defined
3. Estimated effort: ~100 LOC in `train_nnue.py`, no C++ changes needed
4. All infrastructure is already in place (WDL heads, phase heads, loss tracking, logging)

After 3.3, Phase 4 (DCR) becomes the natural next milestone since all its prerequisites will be satisfied.
