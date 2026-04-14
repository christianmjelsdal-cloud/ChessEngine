# Chess Engine — Findings Report
*Generated: 28 March 2026*

---

## 1. Executive Summary

This report documents findings from an investigation into the chess engine training pipeline. Three areas were examined: **best-gen selection logic**, **ELO calibration behaviour**, and a **live training run** (`training_run_20260328_034132.log`). Two previously undetected bugs were identified that together cause `gen_stats.csv` to stop updating after the first training campaign ends, and cause all per-gen elapsed-time logging to be recorded as zero.

---

## 2. Best-Gen Determination

### 2.1 Selection Criterion

Best gen is chosen purely by **lowest validation loss**. There is no ELO component in the selection.

### 2.2 Data Sources (Dual-Source Merge)

`findBestGenFor()` merges two sources before picking the minimum:

| Source | Location | Purpose |
|--------|----------|---------|
| `gen_stats.csv` | `<dataDir>/assets/gen_stats.csv` | Persistent record across restarts |
| `g_st.pts` (in-memory) | TrainingState struct | Live data for the current run |

The merge uses a `std::map<int, double>` keyed by gen number. If the same gen appears in both sources, the lower val_loss wins.

### 2.3 SWA Interaction

Before a gen's val_loss is committed to `gen_stats.csv`, an SWA weight comparison is performed:

1. SWA-averaged weights play a short match against the best-val-checkpoint weights.
2. If SWA wins, its weights **overwrite** `nnue_weights_gen<N>.bin`.
3. The val_loss written to `gen_stats.csv` is always from the **original training checkpoint**, not the SWA model.

Consequence: best-gen selection is val-loss driven even when SWA promotes a different weight file. The val_loss metric and the weights on disk can represent different model states.

### 2.4 Weight Storage

| Artefact | Path |
|----------|------|
| Per-gen weights | `<dataDir>/nnue_weights_gen<N>.bin` |
| Val-loss record | `<dataDir>/assets/gen_stats.csv` |
| Runtime points | `g_st.pts` (in memory, cleared on new pipeline start) |

---

## 3. ELO Calibration (`find_elo.ps1`)

### 3.1 Starting Point

```
last_confirmed_elo.txt  →  exists?  →  use saved ELO as anchor
                               ↓ no
                          $DefaultElo (1500)
```

The script anchors to the last *verified* ELO. If the file is absent (first ever run, or manually deleted), it starts from 1500.

### 3.2 Calibration Phases

| Phase | Games | Condition | Purpose |
|-------|-------|-----------|---------|
| Cal Round 1 | 50 | Always | Initial estimate vs Stockfish |
| Cal Round 2 | 50 | Only if Round 1 adjustment > 50 ELO | Refine large jumps |
| Verify Round 1 | 100 | Always | Confirm estimate |
| Verify Round 2 | 100 | Always | Second confirmation |

### 3.3 Save Logic

- If **both** verify diffs are within **±30 ELO**: result saved to `last_confirmed_elo.txt` ✅
- If either verify diff exceeds ±30: result saved to `last_estimate_elo.txt` only — the confirmed file is **never overwritten with a noisy result**

### 3.4 Observed Behaviour

The script correctly re-anchors to the old confirmed ELO on restart. Large improvement jumps between campaigns are handled by Round 2 refinement. The conservative save policy is intentional and sound.

---

## 4. Training Run Analysis

**Run:** `training_run_20260328_034132.log`  
**Planned:** Gens 7–16 (10 gens) | **Completed:** Gens 7, 8, 9 | **Stopped:** Manually at gen 10 selfplay

### 4.1 Per-Gen Summary

| Gen | Train Loss | Val Loss | Δ Val | SWA Result | Weights Kept |
|-----|-----------|---------|-------|-----------|-------------|
| 7 | 0.1645 | 0.33864 | — | −41 ELO ❌ | Best-val |
| 8 | 0.1627 | **0.33532** | −0.00332 | +6 ELO ✅ | SWA promoted |
| 9 | 0.1611 | 0.33560 | +0.00028 | −48 ELO ❌ | Best-val |

**Gen 8 is the true best** (lowest val_loss = 0.33532, SWA also confirmed better).

### 4.2 Key Observations

**Val loss plateau**
The gap between gen 8 and gen 9 is only 0.00028 — negligible. Training loss continues dropping (model is learning) but validation loss is flat. The model is over-fitting to self-play data rather than generalising.

**Large and stable train/val gap**
~0.175 gap across all three gens. This is a persistent generalisation deficit, not session-level noise. Likely causes: insufficient game diversity, too-low weight decay, or the validation set drifting away from current self-play distribution.

**SWA is inconsistent and mostly harmful**
- Gen 7: −41 ELO
- Gen 8: +6 ELO
- Gen 9: −48 ELO

2 of 3 SWA averages hurt. Root cause is likely premature averaging: with `swaStart=5` and 12 epochs, SWA only covers the last 7 epochs. If the LR cosine schedule has not settled the model by epoch 5, the early snapshots anchor the average in a poor region.

**Opening weakness is consistent**
Across all gens: Opening loss ~0.350, Middlegame ~0.332, Endgame ~0.330. The model consistently under-performs on opening positions.

**Accuracy improving despite val plateau**
Move accuracy rose from 0.577 → 0.613 across the three gens — a positive signal that learning is occurring, just not reflected in val_loss.

### 4.3 Recommendations

| Issue | Suggested Fix |
|-------|--------------|
| Val loss plateau | Increase games per gen (from 750); diversify opening book |
| Train/val gap | Increase weight decay or dropout slightly |
| SWA hurting | Raise `swaStart` to 8–9 (let LR schedule settle first) |
| Opening weakness | Increase SPL ratio for openings or add curated opening positions |

---

## 5. Bug Report

### Bug 1 — `gen_stats.csv` Not Updated After First Campaign *(Critical)*

**File:** `TR_Config.cpp`, `saveGenStat()`, lines 60–68  
**Symptom:** Gen 3 reported as best gen despite gens 7–9 having lower val_loss. Gens 7–9 completely absent from `gen_stats.csv`.

**Root Cause:**

`FIX 6.10` (introduced during the architecture refactor) rewrites `gen_stats.csv` atomically via a temporary file:

```cpp
std::string tmpPath = path + ".tmp";
{
    std::ofstream out(tmpPath, std::ios::trunc);
    for (auto& [g, v] : stats)
        out << g << "," << std::fixed << std::setprecision(8) << v << "\n";
}
// Atomic rename: replace real file with temp (FIX 6.10)
std::rename(tmpPath.c_str(), path.c_str());   // ← BUG
```

On **POSIX/Linux**, `rename()` atomically replaces the destination. On **Windows**, `rename()` (which wraps `MoveFileW`) **fails and returns nonzero if the destination file already exists**. The return value is ignored, so the failure is silent.

**Timeline:**
- Gens 1–6: Written by the old code (simple append mode) — all present in `gen_stats.csv` ✅
- Gen 7 onward: New code tries to rename `.tmp` over existing `gen_stats.csv` → Windows rejects it → `.tmp` file is left on disk, original file unchanged ❌

**Impact:** `find_elo.ps1` and the TrainingRunner UI both pick gen 3 (best of the old run) and use wrong weights indefinitely.

**Fix:**

Replace the `std::rename` call with a Windows-safe replacement:

```cpp
// Windows: rename() fails if destination exists; delete first
std::remove(path.c_str());
if (std::rename(tmpPath.c_str(), path.c_str()) != 0) {
    // Fallback: direct copy if rename still fails (cross-device edge case)
    std::ifstream  src(tmpPath, std::ios::binary);
    std::ofstream  dst(path,    std::ios::binary | std::ios::trunc);
    dst << src.rdbuf();
    std::remove(tmpPath.c_str());
}
```

Or use the Windows API directly for true atomicity:

```cpp
#ifdef _WIN32
    MoveFileExW(
        std::filesystem::path(tmpPath).wstring().c_str(),
        std::filesystem::path(path).wstring().c_str(),
        MOVEFILE_REPLACE_EXISTING);
#else
    std::rename(tmpPath.c_str(), path.c_str());
#endif
```

---

### Bug 2 — `elapsed_sec` Always Logged as 0.0 *(Low)*

**File:** `TR_Pipeline.cpp`, lines 1068–1087  
**Symptom:** All `GEN_SUMMARY` log events show `elapsed_sec=0.0`.

**Root Cause:**

The EMA branch updates `prevGenCompletedSec` to equal `lastGenCompletedSec` **before** the `genSec` calculation reads both values:

```cpp
// Line 1068 — EMA branch (inside lock):
long long thisGenSec = g_st.lastGenCompletedSec - g_st.prevGenCompletedSec; // correct here

// Line 1079 — still inside same lock block:
g_st.prevGenCompletedSec = g_st.lastGenCompletedSec;  // prev is now == last

// Line 1085 — outside lock, for logGenSummary:
double genSec = static_cast<double>(
    g_st.lastGenCompletedSec - g_st.prevGenCompletedSec);  // always 0!
```

`thisGenSec` on line 1068 is computed correctly but only used for the EMA. The separate `genSec` variable on line 1085 (used for logging) recomputes the same subtraction after `prevGenCompletedSec` has already been set equal to `lastGenCompletedSec`, giving zero every time.

**Impact:** Per-gen timing is always zero in log files and any downstream tooling that reads `elapsed_sec`.

**Fix:**

Capture `genSec` before updating `prevGenCompletedSec`:

```cpp
// Inside the lock, before the prevGenCompletedSec update:
long long genSecRaw = g_st.lastGenCompletedSec - g_st.prevGenCompletedSec;
g_st.prevGenCompletedSec = g_st.lastGenCompletedSec;
// ... rest of lock block ...

// Then use genSecRaw for logging (no re-computation needed):
double genSec = static_cast<double>(genSecRaw);
if (genSec < 0) genSec = 0;
g_fileLog.logGenSummary(gen, g_st.lastTrain, g_st.lastVal, g_st.lastElo, genSec);
```

---

## 6. Immediate Action Required

Before the next training run, manually patch `gen_stats.csv` to include the completed gens so the correct weights are used:

```
7,0.33864000
8,0.33532000
9,0.33560000
```

Add these three lines to `<dataDir>/assets/gen_stats.csv`. Gen 8 (val_loss 0.33532) will then be selected as best, and `find_elo.ps1` will use the correct weight file.

---

## 7. Summary Table

| # | Area | Severity | Status |
|---|------|----------|--------|
| 1 | Best-gen uses val_loss only (no ELO factor) | Informational | By design |
| 2 | SWA weights diverge from recorded val_loss | Informational | By design |
| 3 | `gen_stats.csv` silent write failure on Windows (`std::rename`) | **Critical** | Fix pending |
| 4 | `elapsed_sec` always 0.0 in GEN_SUMMARY log events | Low | Fix pending |
| 5 | Val loss plateau after gen 8 | Training concern | Manual tuning needed |
| 6 | SWA hurting 2/3 gens | Training concern | Raise `swaStart` |
| 7 | Opening loss consistently higher than mid/endgame | Training concern | Data diversity |
