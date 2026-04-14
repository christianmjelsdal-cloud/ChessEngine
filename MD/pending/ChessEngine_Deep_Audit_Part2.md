# Chess Engine — Deep Continued Audit (Part 2)
## Line-by-Line Efficiency & Effectiveness Analysis

**Auditor:** Tasklet AI  
**Date:** 18 Mar 2026  
**Codebase:** ~22,800 lines (C++17, Python 3, PowerShell)  
**Hardware:** AMD Ryzen 7 5800U (Zen 3, 8C/16T), 16GB DDR4-3200 CL22, HP Laptop  
**Scope:** Everything the initial audit did NOT cover, plus deeper analysis of previously flagged areas

---

## Table of Contents

1. [Search Algorithm — Deep Dive](#1-search-algorithm--deep-dive)
2. [Move Generation — Staging & Correctness](#2-move-generation--staging--correctness)
3. [Board Representation — Make/Unmake Efficiency](#3-board-representation--makeunmake-efficiency)
4. [NNUE — Forward Pass & Accumulator Deep Dive](#4-nnue--forward-pass--accumulator-deep-dive)
5. [Thread Safety & Lazy SMP Analysis](#5-thread-safety--lazy-smp-analysis)
6. [Time Management Analysis](#6-time-management-analysis)
7. [Duck Chess — Algorithmic Efficiency](#7-duck-chess--algorithmic-efficiency)
8. [Training Pipeline — Deep Dive](#8-training-pipeline--deep-dive)
9. [Self-Play Data Quality](#9-self-play-data-quality)
10. [Memory Layout & Cache Analysis](#10-memory-layout--cache-analysis)
11. [Stack Usage & Crash Risk](#11-stack-usage--crash-risk)
12. [Hardware-Specific Deep Dive (Your Ryzen 7)](#12-hardware-specific-deep-dive)
13. [Missing Modern Techniques (Elo Roadmap)](#13-missing-modern-techniques-elo-roadmap)
14. [Correctness Bugs Found](#14-correctness-bugs-found)
15. [Pipeline Automation Gaps](#15-pipeline-automation-gaps)
16. [Prioritized Fix List](#16-prioritized-fix-list)

---

## 1. Search Algorithm — Deep Dive

### 1.1 Missing Staged Move Generation (Critical — ~5-15% NPS)

**Location:** `Engine::search()` lines calling `MoveGen::getLegalMoves(board, moves)` + `orderMoves()`

The engine generates **all** legal moves upfront, scores them all, sorts them all — then iterates. In the vast majority of nodes (~90%+ at depth ≥ 6), a beta cutoff occurs on the first 1-3 moves. The remaining 20-30 generated moves were wasted work.

**What modern engines do:** Staged move generation:
1. **Stage 1:** Try TT move (no generation needed)
2. **Stage 2:** Generate captures only → score with MVV-LVA/SEE → pick best
3. **Stage 3:** Try killer moves (no generation needed)
4. **Stage 4:** Generate quiets → score with history → pick best incrementally

**Impact:** Avoids generating and scoring ~70% of moves at cut nodes. On your 8-core Zen 3, this alone would yield **5-15% NPS improvement** — likely +10-25 Elo.

**Implementation sketch:**
```cpp
class MovePicker {
    enum Stage { TT_MOVE, GEN_CAPTURES, GOOD_CAPTURES, KILLER1, KILLER2, GEN_QUIETS, QUIETS, BAD_CAPTURES };
    Stage stage_ = TT_MOVE;
    Move ttMove_, killer1_, killer2_;
    MoveList captures_, quiets_;
    int capIdx_ = 0, quietIdx_ = 0;
public:
    Move next(Board& board, int ply);  // Returns next move or invalid
};
```

### 1.2 Missing ProbCut (Significant — +15-30 Elo)

**What it is:** Before the main move loop, at non-PV nodes with depth ≥ 5, search captures at reduced depth (depth - 4) with a window of [beta + margin, beta + margin + 1]. If any capture exceeds this, the node is almost certainly a CUT node — prune immediately.

**Why it matters:** ProbCut catches tactical explosions early. Stockfish gains ~20 Elo from ProbCut. Your engine has no equivalent.

```cpp
// ProbCut: reduced-depth capture search
if (!isPV && !inCheck && depth >= 5 && abs(beta) < MATE_SCORE - 100) {
    int probBeta = beta + 200;
    MoveList caps;
    MoveGen::getLegalCaptures(board, caps);
    for (int i = 0; i < caps.count; i++) {
        if (see(board, caps[i]) < 0) continue;
        UndoInfo undo;
        board.makeMove(caps[i], undo);
        int score = -search(board, depth - 4, -probBeta, -probBeta + 1, ply+1, ...);
        board.unmakeMove(caps[i], undo);
        if (score >= probBeta) return score;
    }
}
```

### 1.3 Null Move: No Verification Search (Moderate — +5-10 Elo)

**Location:** `Engine::search()` — null move section

When NMP gets a score ≥ beta, the engine immediately returns `beta`. In zugzwang-prone endgames (e.g., King + Pawns), this causes catastrophic misevaluation.

**Better approach — verification search:**
```cpp
if (nullScore >= beta) {
    // Verify at reduced depth
    if (depth < 12) return beta;  // Trust NMP at low depths
    int verifyScore = search(board, depth - R - 1, beta - 1, beta, ply + 1, false, false, ...);
    if (verifyScore >= beta) return beta;
    // If verify fails, continue with normal search
}
```

### 1.4 IID → IIR Replacement (Minor Simplification, +2-5 Elo)

**Location:** `Engine::search()` — IID block:
```cpp
if (isPV && !(hashMove.from.isValid()...) && depth >= 4) {
    search(board, depth - 2, alpha, beta, ply, false, true, 0, Move{}, inCheck);
```

This does Internal Iterative Deepening — a full depth-2 search to find a hash move. Modern engines use **Internal Iterative Reduction (IIR)** instead: simply reduce depth by 1 when there's no hash move. Much simpler, nearly as effective:

```cpp
// Replace IID with IIR
if (!(hashMove.from.isValid()) && depth >= 4) {
    depth--;  // That's it
}
```

### 1.5 LMR: No Reduction for Captures (Moderate — +5-15 Elo)

**Location:** `Engine::search()` — LMR section only triggers on `isQuiet`:
```cpp
if (depth >= 3 && i >= 3 && isQuiet && !inCheck) {
    reduction = lmrTable_[...];
```

Modern engines also reduce **bad captures** (negative SEE) with smaller reductions. Your engine has SEE pruning at depth ≤ 4, but at depth > 4, bad captures are searched at full depth.

```cpp
// Add after quiet LMR block:
if (depth >= 3 && i >= 3 && isCap && !inCheck) {
    if (see(board, m) < 0) {
        reduction = lmrTable_[depth][i] / 2;  // Half the quiet reduction
    }
}
```

### 1.6 No Razoring (Minor — +3-8 Elo)

At depth 1-2, if static eval is far below alpha, do a qsearch verification. If confirmed, return the qsearch score immediately without trying all moves.

```cpp
if (!isPV && !inCheck && depth <= 2) {
    int razor_margin = 300 + 200 * depth;
    if (staticEval + razor_margin < alpha) {
        int rScore = qsearch(board, alpha, beta, ply, false);
        if (rScore < alpha) return rScore;
    }
}
```

### 1.7 Aspiration Window Re-Search Logic

**Observation:** The comments mention "Graduated aspiration windows (§2.3)" in `getBestMove`. The implementation uses progressively wider windows on fail. This is good and correctly implemented.

**Minor improvement:** After an aspiration fail-low, some engines reduce the search depth by 1 for the re-search to save time. After fail-high, they don't reduce. This can save significant time on positions where eval drops.

### 1.8 Static Eval When TT Has Exact Score

**Location:** `Engine::search()` — `staticEval = evaluate(board, ply)` is computed before checking if TT has an exact match.

When TT has a score from a deeper search, the TT score is more accurate than the static eval. Using TT score for pruning decisions (RFP, futility, NMP) is slightly better:

```cpp
int staticEval;
if (tte.keyMatches(hash) && tte.depth >= depth && tte.flag == TT_EXACT) {
    staticEval = tte.score;  // Use more accurate TT score
} else {
    staticEval = evaluate(board, ply);
}
```

### 1.9 Quiescence Search: Delta Pruning Margin

**Location:** `Engine::qsearch()`:
```cpp
if (standPat + delta + SearchParams::DELTA_PRUNING_MARGIN < alpha) continue;
```

The delta pruning margin is a constant. Modern engines make this adaptive — wider in middlegames, narrower in endgames, since tactical swings are proportional to material on the board.

### 1.10 History Table: Missing Piece-To Indexing

**Location:** `Engine::scoreMove()` and history updates.

Your history table is indexed by `[color][fromSquare][toSquare]` — the "butterfly" scheme. This is correct but lower resolution than **piece-to** indexing (`[piece][toSquare]`), which modern engines prefer because the same piece going to the same square is more predictive regardless of origin.

**Recommended:** Either switch to `[piece][toSquare]` or add it as a supplementary signal:
```cpp
int score = history_[ci][fromSq][toSq];
score += pieceToHistory_[pieceType][toSq];  // Additional signal
```

---

## 2. Move Generation — Staging & Correctness

### 2.1 Full Legal Move Generation Every Node

**Location:** `MoveGen::getLegalMoves()` generates pseudolegal moves then filters illegal ones.

**The issue:** Even at CUT nodes (which are ~90% of all nodes), every single pseudolegal move is generated, legality-tested (make + isInCheck + unmake), scored, and sorted. At an average position with ~35 legal moves, this is:
- 35 pseudo-legal move generations
- 35 makeMove + isInCheck + unmakeMove calls for legality
- 35 scoreMove calls
- An insertion sort over 35 elements

When the TT move (tried first) causes a cutoff, the other 34 moves were completely wasted.

**Fix:** This is the same as §1.1 (staged generation). Additionally, use **pseudo-legal** moves with legality checking only when the move is actually tried:
```cpp
// Instead of: Generate all legal moves, sort, iterate
// Do: Generate pseudo-legal, pick best with lazy scoring, check legality when tried
```

### 2.2 getLegalCaptures: Redundant Legality Check Pattern

**Location:** `MoveGen::getLegalCaptures()` — likely generates pseudo-legal captures then filters for legality. In qsearch, most captures are legal. The legality check (make + isInCheck + unmake) is expensive relative to the information gained.

**Better:** Generate pseudo-legal captures, sort by MVV-LVA, check legality only when the move is about to be searched. If illegal, skip to next.

### 2.3 Pin Detection Efficiency

If the engine detects pins per-move (checking if a move leaves the king in check), it repeats work. Modern engines compute a **pin mask** once per position and use it to quickly filter illegal moves during generation.

---

## 3. Board Representation — Make/Unmake Efficiency

### 3.1 External Hash Update Anti-Pattern

**Location:** After every `board.makeMove(m, undo)`, the engine calls `incrementalHashUpdate(board, m, undo)` as a separate function.

**Issues:**
- Extra function call overhead per move (including qsearch moves — millions per second)
- The hash update requires re-reading move details that `makeMove` already had access to
- Cache: `undo` struct is written by `makeMove`, then read by `incrementalHashUpdate` — potential cache miss if UndoInfo is large

**Fix:** Integrate hash update into `Board::makeMove()` directly. The Zobrist XOR operations should be interleaved with the board mutation operations while the relevant data is already in registers:
```cpp
void Board::makeMove(const Move& m, UndoInfo& undo) {
    // ... existing board updates ...
    hash ^= zPiece[movedPiece][fromSq];  // Remove piece from origin
    hash ^= zPiece[movedPiece][toSq];    // Place piece at destination
    hash ^= zSide;                       // Toggle side
    // ... etc
}
```

### 3.2 Accumulator Update: Separate Function Call

Same pattern — `updateAccumulator()` is called separately after `makeMove()` and `incrementalHashUpdate()`. That's **three function calls per move** for what is logically one operation. Combining them (or at least inlining) would save function call overhead at millions of calls per second.

### 3.3 UndoInfo Size

The `UndoInfo` struct needs to store everything `unmakeMove` needs to restore. If this struct is large (castling rights, EP square, half-move clock, captured piece, hash, etc.), it adds stack pressure. Check if it can be minimized.

---

## 4. NNUE — Forward Pass & Accumulator Deep Dive

### 4.1 Full Accumulator Copy for Null Move (4 KB Wasted)

**Location:** `Engine::search()` — null move section:
```cpp
if (nnue_ && ply + 1 < ACC_STACK_SIZE) {
    accStack_[ply + 1] = accStack_[ply];  // Full copy
}
```

`accStack_` entries are `L1_SIZE * 2 sides * sizeof(int16_t) = 1024 * 2 * 2 = 4,096 bytes`. This copies 4 KB for every null move attempt.

**Better approach:** Use a pointer/index scheme. Since null moves don't change any pieces, the child can simply reference the parent's accumulator:
```cpp
// Option A: Lazy copy flag
accStack_[ply + 1].copyFrom = ply;  // Resolve lazily if needed

// Option B: Accumulator pointer stack
int16_t* accPtr_[MAX_PLY];  // Points to actual accumulator data
// For null move: accPtr_[ply+1] = accPtr_[ply];  // Just copy pointer
```

### 4.2 Accumulator Copy on Every Normal Move

**Location:** `Engine::updateAccumulator()` — on non-special (non-king, non-castling) moves, the function copies the parent accumulator and then applies incremental add/sub.

The copy is 4 KB. With 16-byte AVX2 registers processing 16 int16_t values at once, that's 256 AVX2 operations just for the copy, plus another 256 for the add/sub updates.

**Better:** Combine copy and update in a single fused loop:
```cpp
// Instead of: memcpy(child, parent, 4KB); then add/sub
// Do: for each chunk: load parent, add delta, store child
for (int i = 0; i < L1_SIZE; i += 16) {
    __m256i parent = _mm256_load_si256(parentAcc + i);
    __m256i delta  = _mm256_load_si256(deltaVec + i);
    _mm256_store_si256(childAcc + i, _mm256_add_epi16(parent, delta));
}
```
This halves memory bandwidth (one read + one write instead of two reads + one write + two reads + one write).

### 4.3 L2 Forward Pass: Row-Major vs Column-Major

**From initial audit — deeper analysis:** The L2 layer (1024 → 256) involves a matrix multiply with weight matrix `w2[L2_SIZE][L1_SIZE * 2]`. Accessing pattern:
```
for each output neuron j (0..255):
    for each input i (0..2047):
        sum += w2[j][i] * input[i]
```

With `w2` stored as `[L2_SIZE][L1_SIZE*2]` (row-major), the inner loop accesses `w2[j][0], w2[j][1], ... w2[j][2047]` which is sequential — **this is actually cache-friendly** for the weight access. The input vector `input[i]` is the same for all j, so it should stay in L1 cache.

**However:** The issue is that with L2_SIZE=256, the full weight matrix is `256 * 2048 * sizeof(int16_t) = 1 MB`. Your Ryzen 7 5800U has 512 KB L2 per core. So the weight matrix alone busts L2 cache.

**Fix: Tiled (blocked) computation:**
```cpp
// Process in tiles of 32 output neurons × 128 inputs
for (int jo = 0; jo < L2_SIZE; jo += 32) {
    for (int io = 0; io < inputSize; io += 128) {
        for (int j = jo; j < jo + 32; j++) {
            for (int i = io; i < io + 128; i++) {
                output[j] += w2[j][i] * input[i];
            }
        }
    }
}
```
This ensures the working set (~32 × 128 × 2 bytes = 8 KB weights + 128 × 2 bytes input = 256 bytes) fits in L1 cache.

### 4.4 NNUE Eval Cache

**Missing entirely.** The same position can be reached via different move orders. Each evaluation recomputes the full L2→L3→output forward pass. Adding an eval hash table (4-8 MB, indexed by accumulator checksum) would cache ~90% of evaluations in the endgame.

```cpp
struct EvalCacheEntry {
    uint64_t accHash;  // Hash of accumulator state
    int16_t score;
};
EvalCacheEntry evalCache_[1 << 20];  // 1M entries = ~12 MB
```

### 4.5 ClippedReLU: Branch-Free Alternative

**Location:** `NNUE::forward()` — The SCReLU (Squared Clipped ReLU) implementation uses `_mm256_max_epi16` and `_mm256_min_epi16`, which is correct and efficient. No issue here — well done.

### 4.6 Quantization Constants

**Observation:** `QA = 255`, `QB = 64`. This gives an accumulator range of [-32768, 32767] which is the full int16 range. However, with L1_SIZE=1024 and ~30 active features, accumulator values will typically be in the range [-7650, 7650] (30 × 255). This means the int16 range is only ~23% utilized, wasting precision.

**Consideration:** Higher QA (e.g., 512) would improve precision at the cost of needing int32 accumulators for the forward pass. This is a design tradeoff — the current choice prioritizes speed (int16 SIMD) over precision.

---

## 5. Thread Safety & Lazy SMP Analysis

### 5.1 TT Race Condition (Known, Acceptable)

Lazy SMP shares the TT between threads without locks. This means:
- Thread A writes a TT entry while Thread B reads it → torn read
- The `keyMatches()` method (XOR key with stored key) mitigates this by detecting corrupted entries

**Your implementation uses `keyMatches()`** which is the standard Stockfish approach. Good.

### 5.2 History Table Contention

**Location:** `history_[2][64][64]`, `killers_[MAX_PLY][2]`, `countermoves_[2][64][64]`, `counterMoveHist_[12][64][64]`

These are **per-engine-instance** (not per-thread). Since Lazy SMP uses multiple Engine instances, each with their own tables, there's no contention. ✅ Correct.

### 5.3 Node Counter: Relaxed Atomics

**Location:** `nodes_` is a regular integer, accumulated with `nodes_++`. In Lazy SMP, each thread has its own Engine instance with its own `nodes_`. The main thread aggregates after search. This is correct.

### 5.4 `stop_` Flag: Memory Ordering

**Location:** `stop_.store(false, std::memory_order_relaxed)` and `stop_.load(std::memory_order_relaxed)`

Relaxed ordering means the stop signal may take a few hundred nanoseconds to propagate to other cores. This is fine — a few extra nodes searched is negligible. ✅ Correct.

### 5.5 accumulator Stack: Per-Thread Isolation

`accStack_` is per-Engine-instance, so each SMP thread has its own stack. No sharing issues. ✅

---

## 6. Time Management Analysis

### 6.1 Soft/Hard Limit Structure

**Location:** `Engine::getBestMove()` uses `softLimitMs_` and `hardLimitMs_`.

- Soft limit: Start new iteration only if elapsed < soft limit
- Hard limit: Abort search mid-iteration

This is a standard and correct approach.

### 6.2 Best Move Instability Extension

**Observation:** The comments mention "extends time when best move is unstable." This is critical — if the best move changes between iterations, more time should be allocated. Need to verify this is actually implemented in the iterative deepening loop.

**If missing:** When the root best move changes, multiply remaining soft limit by 1.5-2.0x. When the root score drops significantly, multiply by 2.0x. This is one of the highest-impact time management features.

### 6.3 Missing: Move Overhead Safety

UCI protocol includes a "Move Overhead" option to account for communication delay. If not implemented, the engine may flag on time in fast time controls (bullet/blitz). On your system, USB/serial latency + Windows scheduler jitter could be 5-50ms.

### 6.4 Missing: Ponder (Thinking on Opponent's Time)

No ponder support was found. When the opponent is thinking, your engine is idle. Pondering is worth **+30-60 Elo** in timed games.

**Implementation:** After returning best move, predict opponent's reply (most likely = PV[1]) and start searching. If opponent plays the predicted move, continue searching. If they play a different move, restart.

---

## 7. Duck Chess — Algorithmic Efficiency

### 7.1 Board Copy at Root (1500 copies/iteration)

**Location:** `Engine::getBestMove()` — Duck chess root search:
```cpp
for (int i = 0; i < chessMoves.count; i++) {
    Board temp = board;              // Copy #1 (~200+ bytes)
    temp.applyMove(chessMoves[i]);
    for (int d = 0; d < duckSquares.count; d++) {
        Board duckBoard = temp;      // Copy #2 (~200+ bytes)
        duckBoard.placeDuck(duckSquares[d]);
```

With ~30 chess moves × ~50 duck placements = **1,500 board copies per iteration** at the root. Over 10+ depth iterations, that's 15,000+ copies.

**Note:** The inner `searchDuck` already uses in-place duck placement with manual undo (good fix from prior audit). But the root search still copies.

**Fix:** Use in-place move/unmake at root too:
```cpp
for (int i = 0; i < chessMoves.count; i++) {
    UndoInfo undo;
    board.makeMove(chessMoves[i], undo);
    // ... duck placement loop with in-place placement ...
    board.unmakeMove(chessMoves[i], undo);
}
```

### 7.2 Duck Placement Scoring: 64-Square Scan

**Location:** `Engine::scoreDuckPlacement()`:
```cpp
Square oppKingSq = {-1, -1};
for (int r = 0; r < 8; r++)
    for (int c = 0; c < 8; c++)
        if (board.squares[r][c].type == PieceType::King && board.squares[r][c].color == opponent)
            oppKingSq = {r, c};
```

This scans all 64 squares to find the opponent's king. But Board already tracks king squares: `board.whiteKingSq` / `board.blackKingSq` (from Board.h). Use those directly.

Then the scoring function scans all 64 squares again for piece interactions:
```cpp
for (int r = 0; r < 8; r++)
    for (int c = 0; c < 8; c++) {
        Piece p = board.squares[r][c];
        if (p.isNone() || p.isDuck() || p.color != opponent) continue;
        // Check rook/queen/bishop alignment
```

**Two 64-square scans** per duck placement × ~50 placements × ~30 chess moves = **192,000 square accesses per depth iteration**.

**Fix:** Use bitboards:
```cpp
uint64_t oppRooksQueens = board.pieceBB[opponent][Rook] | board.pieceBB[opponent][Queen];
// Use line attacks from duckSq intersected with oppRooksQueens
```

### 7.3 Duck Chess: No TT Usage

The `searchDuck` function has no transposition table probe or store. Every position is searched from scratch. Adding TT to duck chess would dramatically improve performance.

### 7.4 Duck Chess: No Null Move / LMR

The duck chess search is a plain minimax alpha-beta with no pruning techniques. At the depths you're searching (likely 4-6 in duck chess), NMP and LMR would significantly improve search depth.

---

## 8. Training Pipeline — Deep Dive

### 8.1 No Data Augmentation (Free +5-10% Data)

**Location:** `train_nnue.py` — data loading section.

Chess positions can be **horizontally mirrored** (flip files a↔h, b↔g, etc.) to double the effective dataset. This is free data augmentation that every major NNUE trainer uses.

```python
def mirror_position(features, eval_score, wdl):
    """Mirror a position horizontally (file flip)."""
    mirrored = []
    for feat in features:
        piece_type = feat // 64
        square = feat % 64
        rank, file = square // 8, square % 8
        mirrored_file = 7 - file
        mirrored.append(piece_type * 64 + rank * 8 + mirrored_file)
    return mirrored, eval_score, wdl  # Score doesn't change under mirror
```

### 8.2 Effective Batch Size Too Large

**Location:** `pipeline.ps1` defaults: `--batch-size 8192` + `--grad-accum 4` = effective batch size **32,768**.

For NNUE training with 40,960 features and L1=1024, this is very large. Stockfish's trainer uses effective batch size ~16,384. Larger batches smooth gradients but reduce the number of parameter updates per epoch, potentially under-training.

**Recommendation:** Reduce `--grad-accum` to 2 (effective 16,384) or reduce `--batch-size` to 4096 with `--grad-accum 4` (effective 16,384).

### 8.3 Learning Rate Schedule: Warm Restarts vs Single Decay

**Location:** `pipeline.ps1` passes `--no-cosine-restarts` for single cosine decay.

For iterative self-play training (where each generation adds new data), cosine **with** warm restarts is typically better because each restart allows the model to escape local minima created by the previous data distribution. The pipeline explicitly disables this.

**Recommendation:** Re-enable cosine restarts (`--cosine-restarts`) with T0 matching the epoch count per generation.

### 8.4 SWA Averaging Window

**Location:** `train_nnue.py` — SWA averages **all** snapshots from the entire training run.

If training is 40 epochs but the model converged around epoch 20, the SWA average includes 20 epochs of suboptimal weights. Modern SWA implementations average only the last N snapshots (e.g., last 25%).

```python
# Better: Only average last 25% of snapshots
class SWA:
    def __init__(self, start_pct=0.75):
        self.start_pct = start_pct
    
    def should_snapshot(self, epoch, total_epochs):
        return epoch >= total_epochs * self.start_pct
```

### 8.5 Eval Loss vs WDL Loss Balance

**Location:** `train_nnue.py` — `--lam 0.5` (50% eval + 50% WDL).

This is a reasonable default. However, modern NNUE trainers use **adaptive lambda**: higher eval weight early in training (to learn piece values), shifting toward WDL as training progresses (to learn positional understanding).

```python
# Adaptive lambda (inspired by Stockfish NNUE training)
current_lam = lam_start + (lam_end - lam_start) * (epoch / total_epochs)
# e.g., lam_start=0.8, lam_end=0.3
```

### 8.6 Missing Eval Score Filtering

The training data includes all positions from self-play games, including:
- Positions with scores > 10 pawns (already won/lost — low learning signal)
- Positions in the first 6-8 plies (book moves — no learning signal)
- Positions where the search was terminated early (unreliable scores)

**Recommendation:** Filter training positions:
```python
# Skip positions with extreme evals (low learning signal)
if abs(eval_score) > 3000:  # centipawns
    continue
# Skip positions in first 8 plies
if ply < 8:
    continue
```

### 8.7 Training on CPU Only

**Location:** `train_nnue.py` — runs on CPU via PyTorch.

Your Ryzen 7 5800U has a Vega 8 iGPU which **could** accelerate training via ROCm/DirectML. However, the iGPU only has 512MB VRAM and limited compute. CPU training is actually the right choice for your hardware.

**However:** The training script doesn't use `torch.compile()` (PyTorch 2.0+), which can speed up CPU training by 20-40% through operator fusion and graph optimization:
```python
net = torch.compile(net, backend='inductor')  # Add after model creation
```

---

## 9. Self-Play Data Quality

### 9.1 Fixed Depth Search

**Location:** `pipeline.ps1` — `--depth 5` default.

Depth 5 is extremely shallow. At depth 5, the engine misses many tactical patterns and produces lower-quality evaluations. The training data quality directly limits how strong the engine can become.

**Recommendation:** Use node-based limits instead (e.g., `--nodes 5000`) or time-based limits (e.g., `--movetime 100`). Even depth 7-8 would significantly improve data quality, though at the cost of longer self-play time.

### 9.2 Opening Diversity

The pipeline supports opening books (`--openings openings.txt`). This is critical for data diversity.

**Verify:** That the opening book has sufficient variety (ideally 5000+ positions). A small opening book creates redundant training data.

### 9.3 Self-Play Workers vs Physical Cores

**Location:** `pipeline.ps1` — `--workers 12` default.

Your CPU has 8 physical cores / 16 threads. Using 12 workers on self-play means each worker fights for resources. Since self-play workers do independent searches:

**Recommendation:** Use `--workers 8` (one per physical core) for maximum per-game search quality, or `--workers 14` leaving 2 threads for OS/system overhead.

---

## 10. Memory Layout & Cache Analysis

### 10.1 TT Entry Size and Alignment

**Location:** `Engine.h` — TTEntry struct:
```cpp
struct TTEntry {
    uint64_t key;    // 8 bytes
    int16_t score;   // 2 bytes
    int16_t depth;   // 2 bytes
    TTFlag flag;     // 4 bytes (enum, likely int)
    Move best;       // ? bytes (contains 2 Squares + PieceType + Square)
    uint8_t gen;     // 1 byte
};
```

**Concern:** If `Move` is 16+ bytes and `TTFlag` is 4 bytes, the total TTEntry could be 32+ bytes. This means:
- With 64MB TT: only ~2M entries (64M / 32)
- Each cache line (64 bytes) holds only 2 entries

**Ideal:** Pack TTEntry to 16 bytes (Stockfish uses 10 bytes per entry, 3 per cache line cluster + 2 padding):
```cpp
struct alignas(16) TTEntry {
    uint16_t key16;      // 2 bytes (upper 16 bits of hash)
    int16_t  score;      // 2 bytes
    int16_t  depth;      // 2 bytes  
    uint16_t bestMove;   // 2 bytes (packed: 6 bits from + 6 bits to + 4 bits flags)
    uint8_t  flag;       // 1 byte
    uint8_t  gen;        // 1 byte
    // Total: 10 bytes, padded to 16
};
```

This would **4x** the effective TT capacity at the same memory budget.

### 10.2 Move Struct Size

If `Move` uses full `int` for rank/col fields in `Square`, each Move is likely 20-32 bytes. In MoveList with MAX_MOVES=256, that's 5-8 KB per MoveList — and move lists are created in every node of the search tree.

**Better:** Pack Move into 16 or even 32 bits:
```cpp
// 16-bit packed move: 6 bits from + 6 bits to + 4 bits flags
using PackedMove = uint16_t;
constexpr PackedMove packMove(int from, int to, int flags) {
    return (from & 0x3F) | ((to & 0x3F) << 6) | ((flags & 0xF) << 12);
}
```

### 10.3 accStack_ Memory Footprint

`accStack_[ACC_STACK_SIZE]` with ACC_STACK_SIZE typically 128-256 and each accumulator being 4 KB:
- 128 × 4 KB = **512 KB** per thread
- With 8 SMP threads = **4 MB** just for accumulator stacks

This competes with the L2 cache (512 KB/core on your Zen 3). Active search plies (top 10-15) should fit in L2, but the full stack definitely doesn't.

### 10.4 History Tables Memory

```
history_[2][64][64]          = 2 × 64 × 64 × 4 = 32 KB
countermoves_[2][64][64]     = 2 × 64 × 64 × sizeof(Move)  ≈ 256-512 KB
counterMoveHist_[12][64][64] = 12 × 64 × 64 × 4 = 196 KB
captureHistory_[2][6][64]    = 2 × 6 × 64 × 4 = 3 KB
```

Total: ~500-750 KB per Engine instance. With 8 SMP threads: **4-6 MB**. This is fine.

---

## 11. Stack Usage & Crash Risk

### 11.1 Stack Overflow Risk in Search

Each recursive `search()` call places on the stack:
- `MoveList moves` = 256 × sizeof(Move) ≈ 4-8 KB
- `Move quietsTriedArr[MAX_MOVES]` = 256 × sizeof(Move) ≈ 4-8 KB
- `int scores[MAX_MOVES]` = 256 × 4 = 1 KB
- `UndoInfo undo` = ? bytes
- Various local variables ≈ 100 bytes

**Total per recursion level: ~10-17 KB**

At MAX_PLY = 128: **1.3 - 2.2 MB stack per thread**.

Windows default thread stack size is **1 MB**. **This will crash with a stack overflow.**

Even at a realistic search depth of 40 plies: 40 × 17 KB = 680 KB, dangerously close to the 1 MB limit.

**Critical fix options:**
1. Increase thread stack size: `CreateThread()` with 4MB stack size or `/STACK:4194304` linker option
2. Move `quietsTriedArr` out of the recursive function into the per-ply indexed array in the Engine class:
```cpp
// In Engine.h:
Move quietsTried_[MAX_PLY][128];  // Per-ply, max 128 quiets tracked

// In search():
Move* quietsTriedArr = quietsTried_[ply];  // Zero stack usage
```
3. Reduce MAX_MOVES for quietsTriedArr — tracking 128 is sufficient, 256 is overkill.

### 11.2 qsearch Stack: Safer But Still Deep

`qsearch()` has a lighter frame (no quietsTriedArr), but qsearch can go 20+ plies deep on forced capture sequences. Combined with the main search depth, total recursion could exceed 128 + 20 = 148 levels.

The `if (ply >= ACC_STACK_SIZE - 1) return evaluate(board, ply);` check prevents accumulator overflow, which is good. But the raw stack can still overflow if ACC_STACK_SIZE is close to MAX_PLY.

---

## 12. Hardware-Specific Deep Dive (Your Ryzen 7 5800U)

### 12.1 Power Profile: "Balanced" → "High Performance" (~5-10% free NPS)

**Location:** PC.txt shows `Active power scheme: Balanceret` (Danish for "Balanced").

On "Balanced," Windows dynamically throttles CPU frequency and parks cores. For a chess engine that needs sustained all-core turbo:
1. Open Power Options → Select "High Performance"
2. Or set AMD-specific: `Ryzen Master` → enable PBO (Precision Boost Override)

**Expected gain:** 5-10% sustained NPS due to faster core ramp-up and no frequency throttling.

### 12.2 Thermal Throttling at 83°C

**Location:** PC.txt shows CPU at 83°C.

The Ryzen 7 5800U has a Tjmax of 105°C with throttling starting around 90-95°C. At 83°C during normal usage, multi-threaded chess will push this to **95-100°C**, triggering throttling.

**Recommendations:**
- Clean laptop vents/fans (dust buildup is common on HP laptops)
- Use a laptop cooling pad (~$20, can reduce temps by 5-10°C)
- In BIOS/Ryzen Master, set a manual temperature limit of 90°C (trades max single-thread boost for sustained multi-thread performance)
- Consider undervolting: Ryzen 5800U often runs stable at -25 to -50mV offset, reducing power and heat while maintaining clock speeds

### 12.3 DDR4-3200 CL22: Memory Latency Impact

Your RAM runs at DDR4-3200 CL22-22-22-52. The CAS latency in nanoseconds:
```
CL / (DDR_speed / 2) × 1000 = 22 / 1600 × 1000 = 13.75 ns
```

This is **high** for DDR4-3200 (optimal is CL14-16 = 8.75-10 ns). Every TT probe and NNUE weight access pays this latency tax.

**Mitigation (software):**
- **TT Prefetch** (from initial audit): `_mm_prefetch(&tt_[hash % ttSize_], _MM_HINT_T0)` before making a move. This hides the ~55 ns total memory latency (13.75 ns CAS + row activate + controller overhead).
- Keep hot data (accumulators, history tables) within L2/L3 cache by reducing their size where possible.

### 12.4 Zen 3 Specific: PEXT/PDEP for Sliding Attacks

AMD Zen 3 (Cezanne/5800U) has **fast** PEXT/PDEP (single cycle). Your current magic bitboard implementation uses multiplication-based magic lookup. PEXT-based lookup is:
- Simpler code
- Faster (no multiply, direct table index)
- Smaller tables (no wasted entries from imperfect magics)

```cpp
#ifdef USE_PEXT
uint64_t bishopAttacks(int sq, uint64_t occupied) {
    return bishopTable_[sq][_pext_u64(occupied, bishopMask_[sq])];
}
#else
// Current magic approach
uint64_t bishopAttacks(int sq, uint64_t occupied) {
    return bishopTable_[((occupied & bishopMask_[sq]) * bishopMagic_[sq]) >> bishopShift_[sq]];
}
#endif
```

**Impact:** ~5-10% faster move generation.

### 12.5 Number of SMP Threads

Your CPU has 8 physical cores with SMT (16 threads). For chess engines:
- **8 threads** (physical cores only) gives the best per-thread performance
- **12-14 threads** utilizes SMT for ~20-30% more total throughput
- **16 threads** causes full SMT contention with diminishing returns

**Recommendation:** Default to 8 threads for playing, 14 threads for self-play/analysis where total throughput matters more than latency.

---

## 13. Missing Modern Techniques (Elo Roadmap)

### Priority-ordered list of missing techniques with estimated Elo gains:

| # | Technique | Est. Elo | Effort | Description |
|---|-----------|----------|--------|-------------|
| 1 | Staged Move Generation | +10-25 | High | Generate moves on demand, not all upfront |
| 2 | Syzygy Tablebases | +50-80 | Medium | Perfect play with ≤7 pieces |
| 3 | Pondering | +30-60 | Medium | Think on opponent's time |
| 4 | ProbCut | +15-30 | Low | Reduced-depth capture verification |
| 5 | NNUE Eval Cache | +10-20 | Low | Cache forward pass results |
| 6 | TT Entry Packing (16 bytes) | +10-20 | Medium | 4× more TT entries in same memory |
| 7 | PEXT Sliding Attacks | +5-10 | Medium | Hardware-accelerated attack generation |
| 8 | LMR for Bad Captures | +5-15 | Low | Reduce search depth for losing captures |
| 9 | IIR (replace IID) | +2-5 | Trivial | Simpler, slightly better than IID |
| 10 | Razoring | +3-8 | Low | Pre-search pruning at low depths |
| 11 | NMP Verification Search | +5-10 | Low | Avoid zugzwang misevaluation |
| 12 | Piece-To History | +5-10 | Low | Better move ordering signal |
| 13 | Adaptive LMR | +3-8 | Low | Adjust LMR by node type (expected CUT/ALL) |
| 14 | Multi-Cut | +3-5 | Low | Prune when multiple moves fail high |
| 15 | Contempt Factor | +5-10 | Trivial | Avoid draws against weaker opponents |
| **Total** | | **~170-380** | | |

---

## 14. Correctness Bugs Found

### 14.1 TT Store on Fail-High: Best Move Preservation Logic May Be Wrong

**Location:** `Engine::search()` — TT store at bottom:
```cpp
if (ttFlag == TT_LOWER && tte.keyMatches(hash)
    && (tte.flag == TT_EXACT || tte.flag == TT_UPPER)
    && tte.best.from.isValid()) {
    storedBest = tte.best;  // Preserve old best move
}
```

**Issue:** This preserves the OLD TT entry's best move when storing a LOWER-bound (fail-high) entry. The intention is "the cutoff move may not be the best move for ordering." But this is backwards — on a fail-high, the cutoff move IS the best move for beta-cutoff ordering. The old entry's "best" from an EXACT or UPPER search may be suboptimal for cutoff ordering.

**Risk:** This causes the TT to return a suboptimal hash move at CUT nodes, reducing move ordering quality. The impact is likely -2 to -5 Elo.

**Fix:** Always store `bestMove` (the cutoff move) as the TT best move:
```cpp
tte.best = bestMove;  // Always store the actual best move found
```

### 14.2 Repetition Detection: Game History Scan Range

**Location:** `Engine::search()`:
```cpp
int scanBack = std::min((int)gameHistory_.size(), rootHalfMoveClock_);
for (int i = (int)gameHistory_.size() - scanBack; i < (int)gameHistory_.size(); i++) {
```

**Concern:** `rootHalfMoveClock_` is set at the root of the search. But during the game, the half-move clock is only reset at the root position's value — it doesn't account for pawn moves/captures made during the search tree. This means the scan range may be too wide (scanning positions before the last irreversible move in the game history), but this only wastes time, it doesn't cause incorrect detection.

However: if `gameHistory_` includes hashes from the search tree (not just the game), this would be a bug. **Verify** that `gameHistory_` only contains hashes from actual game moves, not search tree positions.

### 14.3 Draw Score: Contempt

**Location:** `Engine::drawScore()` — likely returns 0.

Returning exactly 0 for draws creates a subtle issue: the engine sees no difference between making a move that leads to a drawn position and one that doesn't. In practice, this can cause the engine to walk into repetition draws against stronger opponents.

**Better:** Return a small contempt-based score:
```cpp
int Engine::drawScore() const {
    // Slight contempt: prefer avoiding draws as the side to move
    return (rootColor_ == board.turn) ? -10 : 10;
}
```

### 14.4 Qsearch: Stand-Pat Score Inconsistency

**Location:** `Engine::qsearch()`:
```cpp
int bestScore = inCheck ? (-MATE_SCORE + ply) : standPat;
```

When in check, `bestScore` starts at a "mated" score. But if all evasions are searched and none improve upon this, the function returns `-MATE_SCORE + ply`. This is correct (checkmate). However, earlier:

```cpp
if (inCheck && moves.empty()) {
    return -MATE_SCORE + ply;
}
```

This early return handles the same case. The `bestScore` initialization is redundant but not harmful. ✅

---

## 15. Pipeline Automation Gaps

### 15.1 No Regression Testing

The pipeline trains Gen N+1 and optionally runs a quick match against Gen N. But it doesn't maintain a **regression baseline**. If Gen N+1 > Gen N > ... > Gen 1, but Gen N+1 is weaker than Gen N-2 (due to training data drift), this isn't detected.

**Fix:** Every K generations, run a match against a fixed reference engine (e.g., Gen 1 or an external engine like Stockfish at a fixed depth/node limit). Track Elo over time.

### 15.2 No Training Data Deduplication

Self-play generates games that may contain duplicate positions (especially from common openings). These duplicates bias the training data toward already-learned positions.

**Fix:** Add a deduplication step between self-play and training:
```python
# Deduplicate by position hash before training
seen = set()
for pos in training_data:
    h = hash(pos.features)
    if h in seen:
        continue
    seen.add(h)
    filtered_data.append(pos)
```

### 15.3 No Checkpoint Recovery

If the pipeline crashes mid-generation (power failure, OOM, etc.), there's no way to resume. The pipeline restarts from scratch.

**Fix:** Write a `pipeline_state.json` after each completed step:
```json
{"current_gen": 3, "completed_step": "training", "weights": "nnue_weights_gen3.bin"}
```
On startup, check for this file and resume from the last completed step.

### 15.4 Training Log: No LR Tracking

**Location:** `training_log.csv` tracks epoch, loss, val_loss, run_id.

It doesn't track the current learning rate. When debugging training issues, knowing the LR at each epoch is essential (especially with cosine schedule + warmup).

---

## 16. Prioritized Fix List

### Tier 1: Critical (Do First — Highest ROI)

| # | Fix | File(s) | Est. Gain | Effort |
|---|-----|---------|-----------|--------|
| 1 | **Stack overflow prevention** — move quietsTriedArr to per-ply member or increase thread stack | Engine.cpp / linker settings | Prevents crashes | 30 min |
| 2 | **Windows Power Plan → High Performance** | OS setting | +5-10% NPS free | 2 min |
| 3 | **TT Prefetch** (from Audit 1, still not done?) | Engine.cpp | +3-5% NPS | 30 min |
| 4 | **Staged move generation** | New MovePicker class | +10-25 Elo | 2-3 days |
| 5 | **Fused accumulator copy+update** | NNUE.cpp | +5-10% eval speed | 2 hours |

### Tier 2: Significant (High Impact, Moderate Effort)

| # | Fix | File(s) | Est. Gain | Effort |
|---|-----|---------|-----------|--------|
| 6 | **ProbCut** | Engine.cpp | +15-30 Elo | 4 hours |
| 7 | **TT entry packing to 16 bytes** | Engine.h/cpp | +10-20 Elo (4× capacity) | 1 day |
| 8 | **Pondering** | Engine.cpp, UCI.cpp | +30-60 Elo | 2-3 days |
| 9 | **Integrate hash update into makeMove** | Board.cpp | +2-3% NPS | 3 hours |
| 10 | **PEXT sliding attacks** (Zen 3 native) | Bitboard.cpp | +5-10% movegen speed | 1 day |

### Tier 3: Moderate (Good Gains, Low Effort)

| # | Fix | File(s) | Est. Gain | Effort |
|---|-----|---------|-----------|--------|
| 11 | **LMR for bad captures** | Engine.cpp | +5-15 Elo | 1 hour |
| 12 | **NMP verification search** | Engine.cpp | +5-10 Elo | 2 hours |
| 13 | **IID → IIR** | Engine.cpp | +2-5 Elo | 15 min |
| 14 | **Razoring** | Engine.cpp | +3-8 Elo | 1 hour |
| 15 | **NNUE eval cache** | NNUE.cpp | +10-20 Elo | 3 hours |
| 16 | **Data augmentation (horizontal mirror)** | train_nnue.py | +better training | 2 hours |
| 17 | **Fix TT best move on fail-high** | Engine.cpp | +2-5 Elo | 15 min |
| 18 | **Contempt factor** | Engine.cpp | +5-10 Elo (vs weaker) | 30 min |

### Tier 4: Polish (Smaller Gains, Quality of Life)

| # | Fix | File(s) | Est. Gain | Effort |
|---|-----|---------|-----------|--------|
| 19 | Self-play depth 5 → 7+ or node-based | pipeline.ps1 | Better data quality | 15 min |
| 20 | Effective batch size 32K → 16K | pipeline.ps1 | Better convergence | 5 min |
| 21 | Pipeline checkpoint recovery | pipeline.ps1 | Reliability | 2 hours |
| 22 | Duck chess: use king sq tracking | Engine.cpp | Perf fix for duck | 15 min |
| 23 | Duck chess: TT in searchDuck | Engine.cpp | Major duck improvement | 3 hours |
| 24 | Pipeline: regression test baseline | pipeline.ps1 | Catch stagnation | 1 day |
| 25 | `torch.compile()` in training | train_nnue.py | +20-40% train speed | 5 min |
| 26 | Laptop cooling / undervolting | Hardware | Prevent throttling | 30 min |
| 27 | Cosine restarts in pipeline | pipeline.ps1 | Better per-gen training | 5 min |
| 28 | Adaptive lambda (eval/WDL blend) | train_nnue.py | Better training signal | 2 hours |

---

## Summary

This deep audit found **28 additional issues** beyond the initial audit's 36 findings, bringing the total to **64 identified improvements**.

**The five highest-impact items you should tackle next:**

1. 🛡️ **Stack overflow risk** — Your search function uses ~10-17 KB per recursion level. At deep searches on Windows (1 MB default stack), this WILL crash. Fix immediately.
2. ⚡ **Staged move generation** — The single largest algorithmic improvement available. Eliminates ~70% of wasted move generation at cut nodes.
3. 🧠 **ProbCut** — Low implementation effort, +15-30 Elo. The best ratio of Elo gained per line of code added.
4. 🔮 **Pondering** — Free Elo in timed games by thinking during opponent's turn.
5. 🖥️ **Power plan + thermal management** — Free 5-10% NPS from your hardware by switching from Balanced to High Performance and managing laptop thermals.

The engine's foundation is solid — clean code, good NNUE integration, working Lazy SMP, and a mature training pipeline. The improvements above would push it from a strong hobbyist engine toward club-level competitive strength.
