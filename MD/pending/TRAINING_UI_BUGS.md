# Training Pipeline — Known Bugs

**Last updated:** April 17, 2026

---

## Bug 1 — NPS Graph Not Generated During Self-Play

**Symptom:** The NPS panel shows a flat horizontal line or nothing at all. NPS does not vary as expected from game to game.

**Root cause (two parts):**

**Part A — `pushPt` deduplication discards NPS points.**  
`AppState::pushPt` replaces any existing point with the same `(gen, step)`. NPS-only points are pushed with `step = N` (1-based sample index). If a training point for the same gen/step already exists from a previous run loaded from the CSV, the NPS point silently replaces it — losing the NPS value because the CSV-loaded point has `hasNps = false`. Conversely if the NPS point is pushed first and training arrives later, the training point overwrites the NPS value.

The fix: merge NPS into an existing point rather than replace wholesale. When a point with matching `(gen, step)` exists and the incoming point has `hasNps = true`, copy only the NPS fields onto the existing point.

**Part B — Live NPS injection during self-play uses wrong gen.**  
The WM_TIMER live-NPS block computes `liveGen = g_cfg.startGen + 1 + g_st.curGen`. This was fixed in a recent commit but is worth confirming. The previous value `curGen + 1` was wrong for any pipeline starting at gen > 1, causing live NPS points to be injected into the wrong gen slot.

**Files:** `TrainingRunner.cpp` → `AppState::pushPt`, WM_TIMER NPS injection block.

---

## Bug 2 — Self-Play ETA Stuck at Last Value (No Countdown)

**Symptom:** The `ETA HH:MM:SS` value shown in the self-play progress line in the output window (e.g. `ETA 00:01:10`) is parsed and stored, but the banner does not count it down. It stays frozen at the last parsed value. When the last progress line arrives just before completion (e.g. `ETA 00:00:16`), the banner freezes on that value.

**Root cause — `selfPlayEtaSec` is never read by the WM_TIMER banner.**  
`TrainingRunner.cpp` has no `selfPlayEtaSec` / `selfPlayEtaStamp` fields in its local `AppState` (these exist in `TR_Types.h`'s `AppState` for the TR_Pipeline path but not in the standalone `TrainingRunner.cpp` struct). Even if it were parsed, the WM_TIMER banner block only injects countdowns for the `isTraining` branch — there is no countdown display for `isSelfPlay`.

The fix:
1. Add `selfPlayEtaSec` and `selfPlayEtaStamp` to `TrainingRunner.cpp`'s local `AppState`.
2. Parse `ETA HH:MM:SS` from `[SelfPlay]` lines and store to those fields (already partially done in `TR_Pipeline.cpp`; needs porting to `TrainingRunner.cpp`).
3. In the WM_TIMER banner, add an `isSelfPlay` branch that calls `countdown(selfPlayEtaSec, selfPlayEtaStamp)` and appends `Self-play done in: ~Xs` to the banner.

**Files:** `TrainingRunner.cpp` → `AppState` struct, self-play RunProc callback, WM_TIMER banner block.

---

## Bug 3 — Epoch/Batch Progress Stuck at `Epoch 0/N` and `Batch total/total`

**Symptom (screenshot):** The log line reads `[Training] Epoch 0/20  Batch 556/556  Training ETA: 2m39s`. Epoch stays at 0 throughout training and the batch shows the total rather than the current batch.

**Root cause — two separate issues:**

**Part A — `curEpoch` only updates from `ParseLoss` which fires at epoch end.**  
`g_st.curEpoch` is set to `pt.step` when a loss line is parsed. Duck training emits loss lines only at the end of each epoch (`Epoch N/M loss=...`). During the epoch, `curEpoch` sits at the previous epoch's value (or 0 for the first epoch). The banner reflects this stale value.

The fix: parse the `Epoch N/M` prefix from batch-level lines (`\r  batch K/TOTAL ...`) to extract the epoch number and update `curEpoch` immediately, not waiting for the full epoch summary.

**Part B — `curBatch` is set to `totalBatches` on the last batch of an epoch, then never resets.**  
The batch parser updates `g_st.curBatch = bn` and `g_st.totalBatches = tb` from lines matching `"batch N/TOTAL"`. At the end of an epoch, the last batch line sets `curBatch = totalBatches`. At the start of the next epoch, no line arrives to reset `curBatch` to 0 or 1 before the next batch starts printing. So the banner shows `Batch 556/556` at the start of every epoch.

The fix: reset `g_st.curBatch = 0` when a new epoch starts (detectable either from a new `Epoch N/M` line or from the epoch summary line at the end of the previous epoch).

**Files:** `TrainingRunner.cpp` → Training phase RunProc callback batch/epoch parser, WM_TIMER countdown injection.

---

## Bug 4 — CPU Utilization Stays ~16% During Duck Training

**Symptom (screenshot):** AMD Ryzen 7 7730U (8 cores / 16 threads) at 16% during duck chess training. Expected: 80–100%.

**Root cause — `std::thread` creation/destruction every batch.**  
Despite the gradient buffer pre-allocation fix, `std::vector<std::thread> trainThreads` is still declared and constructed inside the batch loop:

```cpp
// Inside batch loop — runs 556 times per epoch:
std::vector<std::thread> trainThreads;
trainThreads.reserve(numChunks);
for (int t = 0; t < numChunks; ++t)
    trainThreads.emplace_back([&, t, ...](){ ... });
for (auto& th : trainThreads) th.join();
```

Each iteration spawns and destroys 16 OS threads. Thread creation on Windows takes ~50–200µs per thread. With 16 threads × 556 batches × 20 epochs = **~178,000 thread create/destroy cycles per gen**. At 200µs each that's 35+ seconds of pure thread overhead per gen, dominating actual compute time.

With a small batch (512 samples ÷ 16 threads = 32 samples per thread), each thread does only ~0.1ms of AVX2 work. The thread creation overhead exceeds the work by 2–10×. This explains why CPU never saturates — threads spend most of their time in the kernel scheduler rather than computing.

**Fix — thread pool (persistent worker threads):**  
Replace the per-batch spawn/join with a persistent pool of `numTrainThreads` threads that wake on a barrier/semaphore, process their chunk, then wait again. The work assignment changes per batch but the threads live for the entire training run.

Minimal implementation: use `std::barrier` (C++20) or a `std::mutex` + `std::condition_variable` task queue. The batch loop only needs to post work and wait for completion — no thread creation.

**Alternative quick fix — increase batch size:**  
With `batchSize = 512` and 16 threads, each chunk is only 32 samples. Increasing to `batchSize = 2048` gives 128 samples per thread, making the thread overhead (~200µs) small relative to compute (~2ms). This won't saturate CPU but will immediately improve utilization without a thread pool.

**Also note — `grads` vector cross-thread sum is O(totalParams × numThreads):**  
After all threads complete, the reduction loop sums 16 gradient vectors of 560K floats each — 9M float additions per batch. This is single-threaded and unvectorized. With AVX2 this could be done in-place 8× faster but the current loop is scalar.

**Files:** `NNUETrainer.cpp` → `trainDuck` batch loop, thread management.

---

## Summary Table

| # | Bug | Severity | Fix Effort |
|---|-----|----------|------------|
| 1 | NPS graph: pushPt overwrites NPS fields, wrong liveGen | Medium | Low |
| 2 | Self-play ETA not counted down in banner | Medium | Low |
| 3 | Epoch 0/N and Batch total/total stuck in training log | Medium | Low |
| 4 | CPU ~16%: per-batch thread spawn dominates compute time | High | Medium |

---

## Bug 5 — Gap in Training Loss Line During Active Generation

**Symptom (screenshot):** A visible break in the blue train loss line appears mid-gen while training is in progress. The line continues correctly after the gap. The gap may persist even after the gen completes if NPS points remain interleaved.

**Root cause — NPS-only points (`hasLoss=false`) sort between training points, breaking the line draw loop.**

`pushPt` sorts all points by `(gen, step)` after every insertion. NPS_SAMPLE points are pushed with `hasLoss=false` and `step=N` (1-based, one per epoch slot). Training loss points are also pushed with `step=N` (epoch number). Both share the same step numbering space within a gen.

Because the deduplication check matches on `(gen, step)`, an NPS point at step=3 and a training point at step=3 are considered duplicates — the later arrival overwrites the earlier. But NPS points arrive during self-play (before training) and training points arrive during training. If both are present and the NPS point happens to land at a step where no training point exists yet, the sort interleaves them:

```
sorted pts = [
  {gen=5, step=2, hasLoss=true},   // training point
  {gen=5, step=3, hasNps=true, hasLoss=false},  // NPS point — no loss
  {gen=5, step=4, hasLoss=true},   // training point
]
```

The train line draw loop:

```cpp
for (size_t i=1; i<pts.size(); i++) {
    if (!pts[i].hasLoss || !pts[i-1].hasLoss) continue;  // ← breaks here
    g.DrawLine(...);
}
```

When `pts[i-1]` is the NPS point (`hasLoss=false`), `continue` skips the segment between step 3 and step 4, creating the gap.

**Why it may not self-heal after gen completes:**  
If the training point for step=3 arrives and the dedup replaces the NPS point with the training point, the gap closes. But if the NPS was at step=3 and training has no point at step=3 (epoch numbers don't always match NPS sample numbers exactly), the NPS point stays permanently interleaved and the gap persists.

**Fix — two-part:**

1. **Separate step namespaces for NPS and training points.** NPS points should use a distinct step offset (e.g. `step = -(sampleIndex)` or store in a separate field) so they never conflict with or interleave with training epoch steps in the sort. The NPS graph draws from `hasNps` points regardless of step ordering, so the sort key doesn't need to match epoch numbers.

2. **Fix the train line draw loop to skip non-loss points without breaking continuity.** Change the draw loop to track the last valid loss point's position rather than requiring consecutive array indices to both have `hasLoss=true`:

```cpp
bool started = false; float px_last = 0, py_last = 0;
for (size_t i = 0; i < pts.size(); i++) {
    if (!pts[i].hasLoss) continue;  // skip NPS-only points, don't break line
    float cx = xf((int)i), cy = yf(pts[i].train);
    if (started) g.DrawLine(&trainPen, px_last, py_last, cx, cy);
    px_last = cx; py_last = cy; started = true;
}
```

This is the same pattern already used for the val, accuracy, LR, and NPS lines — only the train line uses the broken two-index pattern. Fix 2 alone resolves the visual gap immediately; fix 1 removes the underlying data model conflict.

**Files:** `TrainingRunner.cpp` → `AppState::pushPt` sort (fix 1), `DrawGraph` train line loop (fix 2). Same issue exists in `TR_Graph.cpp` `DrawGraph` and `SaveGraphPng`.
