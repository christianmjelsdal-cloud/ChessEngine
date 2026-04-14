# NNUE Architecture — Planned Next Steps

## Vision

Build a phase-aware, draw-informed NNUE architecture that:
- Fully utilizes 100M+ generic positions and 10M+ drawn positions
- Produces directly observable per-phase loss signals
- Lays the groundwork for Dynamic Convergence Reversal (DCR)
- Remains quantization-ready for fast inference at runtime

---

## Phase 1 — Foundation: Feature Set & Data Pipeline

**Goal**: Get the right data flowing into the right shape before touching the network.

### 1.1 Adopt HalfKAv2 Feature Encoding
- Implement or verify HalfKAv2 encoding in the data pipeline
- Each position encoded as two sparse vectors (one per king perspective, ~40K features each)
- Confirm the accumulator supports incremental updates for efficient inference

### 1.2 Tag Positions by Game Phase
- Add a phase label to every training position: `opening`, `middlegame`, or `endgame`
- Use material count as the classifier (e.g., >60% material = opening, <25% = endgame)
- This enables per-head routing during training and per-phase loss tracking

### 1.3 Tag Drawn Positions
- Mark all drawn positions from the drawn dataset explicitly
- Track draw source type where possible (stalemate, insufficient material, 50-move, repetition, fortress)
- Assign phase labels — most drawn positions will be endgame; route accordingly

### 1.4 Build a Stratified Sampler
- Training batches should draw proportionally from: generic decisive, generic drawn, and phase-tagged pools
- Configurable ratios — expose these as parameters in train_nnue.py
- Oversample drawn positions in early training epochs before the network is biased toward decisive games

---

## Phase 2 — Network Architecture

**Goal**: Implement the new multi-head, WDL-output architecture.

### 2.1 Widen L1 to 1024
- Current architecture likely uses 256 or 512
- Upgrade to 1024 neurons in L1 (split as 512 per king perspective, concatenated)
- Apply clipped ReLU (0–127 range) throughout for INT8 quantization compatibility

### 2.2 Add WDL Output Head
- Replace single centipawn output with a 3-value softmax: [P(win), P(draw), P(loss)]
- Derive eval from: `score = P(win) - P(loss)`, scaled to centipawns
- Update loss function to cross-entropy against WDL ground truth labels
- Ground truth: derive WDL from game result + eval score blending (standard approach)

### 2.3 Implement Phase-Blended Multi-Head Output
- After the shared body (L1 → L2), branch into three lightweight heads:
  - Opening head: 32 → 1
  - Middlegame head: 32 → 1
  - Endgame head: 32 → 1
- Final eval = weighted blend: `w_op × opening + w_mg × middlegame + w_eg × endgame`
- Blend weights derived from material count at inference time (tapered eval style)
- Each head can also output WDL if desired (three heads × three outputs)

### 2.4 Network Summary

```
Input: HalfKAv2 (~40,960 sparse features)
  │
  └─► L1: 1024 neurons, clipped ReLU  (512 per king perspective, concatenated)
  │
  └─► L2: 64 neurons, ReLU
  │
  └─► L3: 32 neurons, ReLU
         │
         ├─► Opening head    → WDL (3 values)
         ├─► Middlegame head → WDL (3 values)
         └─► Endgame head    → WDL (3 values)

Runtime eval = tapered blend of three head scores → single centipawn value
```

---

## Phase 3 — Training Script Updates (train_nnue.py)

**Goal**: Update the Python training script to support the new architecture and expose the right knobs.

### 3.1 Per-Head Loss Tracking
- Compute and log loss separately for each head every epoch
- Output format already supported by the GUI:
  `Phase loss -> Opening: X.XX  Middlegame: X.XX  Endgame: X.XX`
- No GUI changes needed — the phase graph panel is already built for this

### 3.2 Per-Phase Loss Weighting (DCR Prerequisite)
- Accept per-phase loss weight parameters: `--weight-opening`, `--weight-middlegame`, `--weight-endgame`
- Default all to 1.0 (equal weighting)
- DCR will later adjust these dynamically based on divergence detection

### 3.3 WDL Label Generation
- Add a preprocessing step to convert position results into WDL ground truth
- Standard formula: blend game result (1/0.5/0) with eval-based WDL estimate using a sigmoid
- Configurable blend factor (how much to trust the result vs. the eval)

### 3.4 Drawn Position Oversampling Schedule
- Add `--draw-oversample-epochs N` parameter
- For first N epochs, drawn positions are upweighted by a configurable multiplier
- After N epochs, revert to standard draw ratio

---

## Phase 4 — Dynamic Convergence Reversal (DCR)

**Goal**: Implement the adaptive phase-balancing system described in DCR_Design_Notes.md.

### 4.1 Divergence Detection
- Monitor per-phase loss over a rolling window of N generations
- Compute normalized divergence: each phase's loss deviation from the group average, relative to its own moving average
- Trigger intervention when any phase exceeds a configurable divergence threshold

### 4.2 Continuous Reweighting (Primary Lever)
- When divergence is detected, adjust per-phase loss weights using:
  `weight_phase = (phase_loss / avg_loss) ^ alpha`
- Phases falling behind get stronger gradient signal immediately
- No mode switch needed — takes effect within the same generation

### 4.3 Data Composition Blending (Secondary Lever)
- Gradually shift draw ratio, self-play depth, and game count toward the lagging phase's preferred settings
- Interpolate between preset anchor configs rather than hard-switching
- Change takes effect over the next generation's self-play

### 4.4 GUI Integration
- Add DCR enable/disable toggle to config panel
- Add divergence threshold and rolling window size fields
- Show active DCR state in the banner (e.g., `DCR: → Endgame`)
- Log DCR interventions in the training log with gold color (#FFD700)

### 4.5 Safeguards
- Cooldown: minimum N generations between interventions
- Hysteresis: divergence must persist for M consecutive checks before triggering
- Early training lockout: DCR inactive for first K generations (let baseline establish)
- Cap on maximum weight imbalance to prevent runaway specialization

---

## Phase 5 — Quantization & Inference

**Goal**: Ensure the trained network runs fast at inference time.

### 5.1 INT8 Quantization
- Verify clipped ReLU outputs are within quantization range throughout training
- Add quantization-aware training (QAT) in final epochs if needed
- Export quantized weights in the engine's native format

### 5.2 Tapered Blend at Runtime
- Implement material-count-based blending of the three head outputs in the engine's eval function
- Blend weights should be fast to compute (integer arithmetic)
- Verify blend boundaries feel natural in test games (no discontinuities near phase transitions)

### 5.3 Benchmark
- Compare new architecture vs. current baseline on:
  - Inference speed (nodes per second)
  - ELO gain in self-play
  - Per-phase test loss
  - Draw accuracy (does it correctly evaluate known drawn positions?)

---

## Milestone Summary

| Phase | Deliverable | Dependency |
|---|---|---|
| 1 | Data pipeline with phase tags and drawn position handling | Dataset access |
| 2 | New network architecture (HalfKAv2, WDL, multi-head) | Phase 1 |
| 3 | Updated train_nnue.py with per-phase loss weights | Phase 2 |
| 4 | DCR divergence detection and reweighting | Phase 3 |
| 4 (GUI) | DCR controls and log output in TrainingRunner | Phase 4 logic |
| 5 | Quantization and inference benchmark | Phase 4 complete |

---

## Open Decisions

- **Draw WDL heads**: Should each phase head output its own WDL, or share a single WDL head? (Per-head is more informative but more complex)
- **Blend weight computation**: Precomputed table vs. on-the-fly material count at inference?
- **DCR activation threshold**: What delta in phase loss constitutes "divergence"? Needs empirical tuning
- **Drawn position oversampling multiplier**: Needs testing — too high and the network will bias toward draws
