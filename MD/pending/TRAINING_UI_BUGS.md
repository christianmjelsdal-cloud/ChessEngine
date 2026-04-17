# Training Pipeline — Bug Log

**Last updated:** April 17, 2026  
**Status:** All 5 bugs fixed.

---

## Bug 1 — NPS Graph Not Generated Properly

**Symptom:** NPS panel showed a flat horizontal line or nothing. NPS did not vary as expected from game to game.

**Root cause (two parts):**

**Part A — `pushPt` wholesale replacement discarded NPS fields.**  
`AppState::pushPt` replaced any existing point at the same `(gen, step)` entirely. NPS-only points (`hasNps=true, hasLoss=false`) and training loss points (`hasLoss=true`) share the same step numbering space within a gen. Whichever arrived second silently overwrote the other — NPS data was lost when a training point arrived at the same step, and vice versa.

**Part B — Live NPS injection used wrong gen number.**  
The WM_TIMER live-NPS block computed `liveGen = g_st.curGen + 1`, which was wrong for any pipeline starting at `startGen > 1`. NPS points were injected into the wrong gen slot and never appeared on the graph.

**Fix applied:**
- `pushPt` now merges fields instead of replacing wholesale. If the incoming point has `hasLoss=true`, only loss/val/acc/lr/phase fields are updated. If it has `hasNps=true`, only NPS fields are updated. Both can coexist on the same `(gen, step)` point.
- `lastTrain`/`lastVal` only update when the incoming point has loss data, so an NPS-only push no longer clobbers the displayed last loss values.
- `liveGen` corrected to `g_cfg.startGen + 1 + g_st.curGen`.

**Files fixed:** `TrainingRunner.cpp` → `AppState::pushPt`, WM_TIMER NPS injection block. Same merge logic added to `TR_Types.h` → `AppState::pushPt`.

---

## Bug 2 — Self-Play ETA Frozen (No Countdown)

**Symptom:** The `ETA HH:MM:SS` value from `[SelfPlay]` progress lines was visible in the output window but the banner never counted it down. It froze at the last parsed value (e.g. `ETA 00:00:16`) until self-play completed.

**Root cause:** `TrainingRunner.cpp`'s local `AppState` had no `selfPlayEtaSec` / `selfPlayEtaStamp` fields. The self-play RunProc callback never parsed the ETA from `[SelfPlay]` lines. The WM_TIMER banner block only had a countdown branch for `isTraining` — there was no `isSelfPlay` countdown.

**Fix applied:**
- Added `selfPlayEtaSec` (int) and `selfPlayEtaStamp` (time_point) to `AppState`.
- Self-play RunProc callback now parses `ETA HH:MM:SS` from every `[SelfPlay]` line using `sscanf` and stores it with a timestamp.
- `selfPlayEtaSec` reset to 0 at self-play phase start.
- WM_TIMER banner now reads `spEta`/`spStamp` under lock and appends `Self-play done in: ~Xs` when `isSelfPlay` is active.

**Files fixed:** `TrainingRunner.cpp` → `AppState` struct, self-play RunProc callback, WM_TIMER state snapshot, WM_TIMER banner build block.

---

## Bug 3 — Epoch/Batch Progress Stuck at `Epoch 0/N` and `Batch total/total`

**Symptom:** The live training log line showed `[Training] Epoch 0/20  Batch 556/556  Training ETA: 2m39s` throughout training. Epoch never advanced from 0 during the first epoch, and batch always showed the total count rather than the current batch.

**Root cause (two parts):**

**Part A — `curEpoch` only updated from `ParseLoss`, which fires at epoch end.**  
Duck training emits loss lines only at the end of each epoch (`Epoch N/M loss=...`). During the epoch, `curEpoch` sat at the previous epoch's value (0 for the first epoch). The banner reflected this stale value for the entire epoch duration.

**Part B — `curBatch` was never reset between epochs.**  
The batch parser set `curBatch = bn` from `"batch N/TOTAL"` lines. At the end of each epoch, the last batch line set `curBatch = totalBatches`. At the start of the next epoch, no reset occurred before the first batch line arrived, so the banner showed `Batch 556/556` at the start of every epoch.

**Fix applied:**
- Training RunProc callback now also parses `Epoch N/M` from batch-level lines (not just from loss summary lines). When the epoch number changes, `curEpoch` is updated immediately and `curBatch` is reset to 0.
- This means the banner updates to the correct epoch number as soon as the first batch line of that epoch arrives, and shows `Batch 1/556` rather than `Batch 556/556`.

**Files fixed:** `TrainingRunner.cpp` → Training phase RunProc callback batch/epoch parser.

---

## Bug 4 — CPU Utilization ~16% During Duck Training

**Symptom:** AMD Ryzen 7 7730U (8 cores / 16 threads) at 16% during duck chess training. Expected: 80–100%.

**Root cause — `std::thread` created and destroyed every batch.**  
`std::vector<std::thread> trainThreads` was declared inside the batch loop. Each of 556 batches × 20 epochs spawned and joined 16 OS threads — ~178,000 thread lifecycle events per gen. Thread creation on Windows costs ~50–200µs each. With only 32 samples per thread (512 batch ÷ 16 threads), each thread did ~0.1ms of AVX2 work but ~200µs of kernel overhead. Threads spent more time in the scheduler than computing.

**Fix applied — persistent thread pool:**  
Replaced per-batch spawn/join with a pool of `numTrainThreads` persistent worker threads created once before the epoch loop. Workers sleep on a `std::condition_variable` between batches. Each batch calls `dispatchBatch()` which:
1. Assigns work ranges to `threadWork[]`
2. Zeros the pre-allocated gradient buffers
3. Increments `poolCurEpoch` and calls `poolWake.notify_all()`
4. Waits on `poolDone` until all workers signal completion

Workers are shut down cleanly after the epoch loop via `poolStop = true` + `poolWake.notify_all()`. Zero thread creation overhead across all batches in all epochs.

The forward/backward pass code was moved into the persistent worker lambda (same AVX2 implementation, same correctness).

**Files fixed:** `NNUETrainer.cpp` → `trainDuck` — thread pool setup, `dispatchBatch` helper, worker lambda, pool shutdown.

---

## Bug 5 — Gap in Training Loss Line During Active Generation

**Symptom:** A visible break in the blue train loss line appeared mid-gen while training was in progress. The gap could persist after the gen completed if NPS points remained interleaved between training points.

**Root cause — NPS-only points sorted between training points, breaking the draw loop.**  
`pushPt` sorted all points by `(gen, step)`. NPS_SAMPLE points (`hasLoss=false`, `step=N`) and training loss points (`hasLoss=true`, `step=N`) shared the same step numbering space. After sorting, an NPS point could land between two training points:

```
[{step=2, hasLoss=true}, {step=3, hasNps=true, hasLoss=false}, {step=4, hasLoss=true}]
```

The train line draw loop used a two-index pattern:
```cpp
for (size_t i=1; i<pts.size(); i++) {
    if (!pts[i].hasLoss || !pts[i-1].hasLoss) continue;  // breaks here
    g.DrawLine(...);
}
```
When `pts[i-1]` was the NPS point, `continue` skipped the segment from step 3 to step 4, creating the gap.

**Fix applied:**  
Changed the train line draw loop in all three graph renderers (`TrainingRunner.cpp::DrawGraph`, `TR_Graph.cpp::DrawGraph`, `TR_Graph.cpp::SaveGraphPng`) to the "track last valid point" pattern already used by all other curves:

```cpp
bool st=false; float px_t=0, py_t=0;
for (size_t i=0; i<pts.size(); i++) {
    if (!pts[i].hasLoss) continue;  // skip NPS-only points without breaking line
    float cx=xf((int)i), cy=yf(pts[i].train);
    if (st) g.DrawLine(&trainPen, px_t, py_t, cx, cy);
    px_t=cx; py_t=cy; st=true;
}
```

NPS-only points are now skipped without interrupting the line. The underlying data model conflict (shared step namespace) is partially resolved by Bug 1's merge fix — NPS and loss data now coexist on the same point rather than competing for the same slot.

**Files fixed:** `TrainingRunner.cpp` → `DrawGraph` train line loop. `TR_Graph.cpp` → `DrawGraph` and `SaveGraphPng` train line loops (both instances replaced via PowerShell string replacement).

---

## Summary

| # | Bug | Severity | Status |
|---|-----|----------|--------|
| 1 | NPS graph: `pushPt` wholesale replace discards NPS/loss fields | Medium | ✅ Fixed |
| 2 | Self-play ETA frozen — no countdown in banner | Medium | ✅ Fixed |
| 3 | Epoch stuck at 0, batch stuck at total/total | Medium | ✅ Fixed |
| 4 | CPU ~16%: per-batch thread spawn/join overhead | High | ✅ Fixed |
| 5 | Gap in train loss line from interleaved NPS-only points | Medium | ✅ Fixed |
