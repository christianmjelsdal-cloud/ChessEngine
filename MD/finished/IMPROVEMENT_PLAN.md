# Chess Engine Training Improvement Plan

**Target System:** AMD Ryzen 7 7730U · 16 GB DDR4 · No discrete GPU  
**Current State:** Gen 34 · Val Loss 0.534 · 99% draws in self-play  
**Architecture:** 512 → 128 → 64 → 3 (just downsized from 1024)

---

## Phase 1: Fix the Draw Crisis (Week 1)
> **Goal:** Drop draws from 99% to 40-50% — this is the single biggest training bottleneck.

### 1A. Random Opening Plies ⭐ Highest Impact
- For the first **8 ply** (4 moves per side), select randomly from the **top 3 legal moves** weighted by eval score
- This prevents both sides from playing the same mainline every game
- Implementation: add a `randomPly` counter in self-play; when `ply < 8`, use softmax over top-3 move evals to pick
- **Expected effect:** Draws drop to ~50-60% immediately

### 1B. Draw Adjudication Tuning
| Setting | Current (estimated) | New |
|---------|-------------------|-----|
| Draw eval threshold | ±10 cp | **±5 cp** |
| Consecutive moves required | 4-6 | **12** |
| Min move number for draw adj | low/none | **move 40+** |

### 1C. Resign Adjudication (Add New)
- Adjudicate as a loss when eval is **≤ -800 cp for 5 consecutive moves**
- Ends hopeless games early → more decisive games per hour
- Frees time budget for generating more games

### 1D. Training Settings
| Setting | Current | New | Why |
|---------|---------|-----|-----|
| Draw Weight | 1.5 | **0.5** | De-emphasize drawn positions in loss function |
| Workers | 12 | **10** | Leave headroom for OS + training threads |
| Depth | 7 | **7** (keep for now) | Increase later once draws are fixed |

---

## Phase 2: MultiPV Support (Week 2)
> **Goal:** Enable the engine to search and report the top N best moves.

### 2A. Core MultiPV Implementation
- Add `int multiPV = 1` option in UCI options
- Modify root search loop in `getBestMove()`:
  - After finding the best move for PV line 1, exclude it from the move list
  - Repeat the search for PV line 2, 3, ... up to N
  - Report each line with `info multipv 1 ...`, `info multipv 2 ...`, etc.
- Store an array of `PVLine` structs (move + score + depth + PV moves)

### 2B. UCI Protocol Support
- Add `setoption name MultiPV value N` handling in `UCI.cpp`
- Output `multipv` field in `info` strings
- Respect `multiPV` count in `go` commands

### 2C. Self-Play Integration
- During self-play, use **MultiPV 3**
- Pick from top 3 moves using **softmax temperature selection**:
  - `P(move_i) = exp(score_i / T) / Σ exp(score_j / T)`
  - Temperature `T = 50` cp (tunable) — high T = more random, low T = more greedy
- After move 20, switch back to MultiPV 1 (play best move) to keep game quality
- This replaces/complements the random opening plies from Phase 1

---

## Phase 3: Eval Noise & Data Quality (Week 3)
> **Goal:** Further diversify self-play and improve training data pipeline.

### 3A. Eval Noise Injection
- During self-play search, add **±10-15 cp** random noise to leaf evaluations
- Implementation: at leaf nodes in `evaluate()`, add `(rng() % 30) - 15` when a `noisyEval` flag is set
- This creates slight asymmetry between the two sides even with identical engines
- Disabled during normal play/analysis — only active in self-play mode

### 3B. Opening Book Integration
- Download/create a curated opening book (UHO_XXL or similar EPD/PGN format)
- Parse book positions and use them as self-play starting positions
- Ensures coverage of diverse openings: gambits, closed positions, sharp tactical lines
- Mix 50% book starts + 50% random ply starts for maximum variety

### 3C. Training Data Filtering
- **Cap draw ratio at 45%** per training batch — randomly discard excess drawn games
- Keep **100% of decisive games** (wins and losses)
- Weight endgame positions from decisive games **1.5x** — these teach the engine to convert advantages

---

## Phase 4: Search Depth & Polish (Week 4)
> **Goal:** With draws fixed and data quality improved, push search quality.

### 4A. Increase Self-Play Depth
- Raise depth from 7 → **8 or 9** once draws are under 50%
- Monitor generation time — keep each gen under ~90 minutes
- Deeper search = better labels = stronger training signal

### 4B. FRC/Chess960 Training Mix
- Add **20% Chess960** random start positions to self-play
- Forces the engine to learn general piece coordination rather than memorized openings
- Implementation: random back-rank generator with standard FRC rules

### 4C. ELO Validation Games
- Run periodic validation matches (every 5 gens) against a fixed-strength opponent
- Track ELO progression — training should show steady gains
- If ELO stalls for 10+ gens, trigger LR reduction or architecture review

---

## Implementation Order & Expected Results

```
Phase 1 (Week 1)  →  Draws: 99% → 45-55%    Training quality: ████████░░ 
Phase 2 (Week 2)  →  Draws: 45% → 35-45%     + GUI analysis mode
Phase 3 (Week 3)  →  Data diversity: ██████████ Maximum variety
Phase 4 (Week 4)  →  Search depth ↑           ELO gains accelerate
```

### Estimated Combined Impact
| Metric | Before | After All Phases |
|--------|--------|-----------------|
| Self-play draw rate | 99% | **35-45%** |
| Training data quality | Poor (draw-dominated) | **Balanced & diverse** |
| Eval accuracy | Plateaued | **Steadily improving** |
| Games/hour | ~200 | **250+** (resign adj saves time) |
| NPS | 613K | **613K+** (unchanged, already good) |

---

## Files to Modify

| File | Phase | Changes |
|------|-------|---------|
| `Engine.cpp` | 1, 2, 3 | Random plies, MultiPV root loop, eval noise |
| `Engine.h` | 2, 3 | MultiPV option, noise flag, PVLine struct |
| `UCI.cpp` | 1, 2 | MultiPV option parsing, resign adjudication settings |
| `SelfPlay` (new/existing) | 1, 3, 4 | Opening book loader, game adjudication, FRC starts |
| `NNUETrainer.cpp` | 1, 3 | Draw weight, data filtering, draw ratio cap |

---

*Plan created: March 18, 2026*
