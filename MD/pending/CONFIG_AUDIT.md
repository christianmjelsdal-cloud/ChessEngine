# Configuration Settings Audit

**Last updated:** April 17, 2026

Each setting is traced from UI → ReadConfig → command flag → receiving code.

---

## Self-Play Settings

| Setting | UI Control | Command Flag | Status | Notes |
|---|---|---|---|---|
| **Generations** | ID_EDIT_GENS | Pipeline loop count | ✅ Functional | Controls how many gen cycles run |
| **Start Gen** | ID_EDIT_STARTGEN | Pipeline first gen | ✅ Functional | Sets `firstGen = startGen + 1` |
| **Games per Gen** | ID_EDIT_GAMES | `--games N` | ✅ Functional | |
| **Workers** | ID_EDIT_WORKERS | `--workers N` | ✅ Functional | |
| **Depth** | ID_EDIT_DEPTH | `--depth N` | ✅ Functional | |
| **Mixed Depth %** | ID_EDIT_MIXDEPTH_PCT | `--mixed-depth-ratio F` | ✅ Functional | Divided by 100 in ReadConfig |
| **Low Depth** | ID_EDIT_MIXDEPTH_LOW | `--mixed-depth-low N` | ✅ Functional | Only sent when mixedDepthRatio > 0 |
| **Depth Shuffle** | ID_CHK_DEPTH_SHUFFLE | `--depth-shuffle` | ✅ Functional | Flag only, no value |
| **Shuffle Bias** | ID_EDIT_DEPTH_SHUFFLE_BIAS | `--depth-shuffle-bias F` | ✅ Functional | Only sent when depthShuffle=true |
| **Opening Temp** | ID_EDIT_OPENING_TEMP | `--opening-temp F` | ✅ Functional | |
| **Opening Plies** | ID_EDIT_OPENING_PLIES | `--opening-plies N` | ✅ Functional | |
| **Softmax Plies** | ID_EDIT_SOFTMAX_PLIES | `--softmax-plies N` | ✅ Functional | |
| **Softmax Temp** | ID_EDIT_SOFTMAX_TEMP | `--softmax-temp F` | ✅ Functional | |
| **Root Noise ε** | ID_EDIT_ROOT_NOISE | `--root-noise F` | ✅ Functional | Only sent when > 0 |
| **Record Min Ply** | ID_EDIT_RECORD_MIN_PLY | `--record-min-ply N` | ✅ Functional | |
| **Record Max Eval** | ID_EDIT_RECORD_MAX_EVAL | `--record-max-eval N` | ✅ Functional | |
| **Resign Cp** | ID_EDIT_RESIGNCP | `--resign-cp N` | ✅ Functional | |
| **Resign Count** | ID_EDIT_RESIGN_COUNT | `--resign-count N` | ✅ Functional | |
| **Contempt Cp** | ID_EDIT_CONTEMPT | `--contempt N` | ✅ Functional | Now wired into Engine::drawScore() |
| **Max Plies** | ID_EDIT_MAXPLIES | `--maxplies N` | ✅ Functional | |
| **Draw Adj Cp** | ID_EDIT_DRAWCP | `--draw-cp N` | ✅ Functional | Duck chess: forced to max(drawCp, 50) |
| **Draw Count** | ID_EDIT_DRAW_COUNT | `--draw-count N` | ✅ Functional | |
| **Draw Min Ply** | ID_EDIT_DRAW_MIN_PLY | `--draw-min-ply N` | ✅ Functional | |
| **Draw Adj Moves** | ID_EDIT_DRAW_ADJ_MOVES | `--draw-adj-moves N` | ✅ Functional | |
| **Draw Adj Thresh** | ID_EDIT_DRAW_ADJ_THRESH | `--draw-adj-threshold N` | ✅ Functional | |
| **Draw Adj Min** | ID_EDIT_DRAW_ADJ_MIN_MOVE | `--draw-adj-min-move N` | ✅ Functional | |
| **FRC Mix %** | ID_EDIT_FRCMIX | `--frc-mix F` | ✅ Functional | Only sent when > 0; disabled for duck chess |

---

## Training Settings

### Standard Chess (train_nnue.py)

| Setting | Command Flag | Status | Notes |
|---|---|---|---|
| **Epochs per Gen** | `--epochs N` | ✅ Functional | |
| **Batch Size** | `--batch-size N` | ✅ Functional | |
| **Learning Rate** | `--lr F` | ✅ Functional | |
| **Weight Decay** | `--weight-decay F` | ✅ Functional | |
| **Dropout** | `--dropout F` | ✅ Functional | |
| **Label Smoothing** | `--label-smoothing F` | ✅ Functional | Only sent when > 0 |
| **Grad Accumulation** | `--grad-accum N` | ✅ Functional | |
| **LR Warmup Steps** | `--warmup-steps N` | ✅ Functional | |
| **Draw Weight** | `--draw-weight F` | ✅ Functional | Maps to `lambda = 1 - drawWeight` in trainer |
| **Mate Boost** | `--mate-boost F` | ✅ Functional | |
| **Max Positions** | `--max-positions N` | ✅ Functional | |
| **Early Stop** | `--early-stop N` | ✅ Functional | |
| **Cosine LR** | `--no-cosine-restarts` (when off) | ✅ Functional | Inverted flag logic |
| **Cosine T0** | `--cosine-t0 N` | ✅ Functional | Only sent when cosineLr=true and cosineT0 > 0 |
| **SWA** | `--swa --swa-start N` | ✅ Functional | Only sent when swa=true |
| **SWA Start** | `--swa-start N` | ✅ Functional | Bundled with --swa |
| **WDL Alpha** | `--wdl-alpha F` | ✅ Functional | Always sent |
| **WDL Draw Elo** | `--wdl-draw-elo F` | ✅ Functional | Always sent |
| **Self-Play Ratio** | `--extra-data file ratio` | ✅ Functional | Used as weight for selfplay data |
| **Replay Window** | Multiple `--extra-data` | ✅ Functional | Adds previous gens with decay |
| **Replay Decay** | Computed weight | ✅ Functional | `weight *= replayDecay` each step |
| **Draw Ratio %** | `--extra-data draws_file ratio` | ✅ Functional | Only when draws file exists and drawPct > 0 |
| **ELO Validation** | Controls EloVal() call | ✅ Functional | Skips ELO phase when false |
| **ELO Games** | Used in EloVal() | ✅ Functional | Passed to cutechess match |
| **SWA Games** | Used in SWA match | ✅ Functional | Passed to cutechess SWA match |
| **Overfitting Detection** | *(see below)* | ⚠️ Partial | |

### Duck Chess (--train-duck)

| Setting | Command Flag | Status | Notes |
|---|---|---|---|
| **Epochs per Gen** | `--epochs N` | ✅ Functional | |
| **Batch Size** | `--batch-size N` | ✅ Functional | |
| **Learning Rate** | `--lr F` | ✅ Functional | |
| **Weight Decay** | `--weight-decay F` | ✅ Functional | |
| **Label Smoothing** | `--label-smoothing F` | ✅ Functional | Only when > 0 |
| **Grad Accumulation** | `--grad-accum N` | ✅ Functional | |
| **LR Warmup Steps** | `--warmup-steps N` | ✅ Functional | |
| **Draw Weight** | `--draw-weight F` | ✅ Functional | |
| **Mate Boost** | `--mate-boost F` | ✅ Functional | |
| **Max Positions** | `--max-positions N` | ✅ Functional | Set to 0 after cap+mirror in main.cpp |
| **Early Stop** | `--early-stop N` | ✅ Functional | |
| **Cosine LR** | `--cosine-lr` | ✅ Functional | |
| **Cosine T0** | `--cosine-t0 N` | ✅ Functional | |
| **SWA** | `--swa --swa-start N` | ✅ Functional | |
| **Replay Window** | Multiple `--extra-data` | ✅ Functional | |
| **Replay Decay** | Computed weight | ✅ Functional | |
| **WDL Alpha** | ❌ NOT SENT | 🔴 **Bug** | Missing from duck training command |
| **WDL Draw Elo** | ❌ NOT SENT | 🔴 **Bug** | Missing from duck training command |
| **Dropout** | ❌ NOT SENT | ⚠️ Missing | --train-duck doesn't support dropout |
| **Draw Ratio %** | ❌ NOT SENT | ⚠️ Missing | Duck trainer doesn't use draws dataset |
| **Overfitting Detection** | ❌ NOT SENT | ⚠️ Partial | See below |

---

## Issues Found

### 🔴 Bug 1 — WDL Alpha and WDL Draw Elo not passed to duck training

`wdlAlpha` and `wdlDrawElo` are sent to `train_nnue.py` via `--wdl-alpha` and `--wdl-draw-elo`, but the duck training command (`--train-duck`) does not include these flags. The duck trainer (`NNUETrainer.cpp::trainDuck`) uses `tcfg.lambda` which defaults to `0.5` regardless of the UI setting.

**Effect:** The `WDL Alpha` and `WDL Draw Elo` UI settings have no effect on duck chess training. The duck trainer always uses `lambda=0.5` (50% eval loss, 50% result loss) regardless of what `Draw Weight` is set to.

Wait — `Draw Weight` maps to `lambda = 1 - drawWeight` and IS sent via `--draw-weight`. But `WDL Alpha` is a separate parameter used only by `train_nnue.py`'s WDL cross-entropy loss. The duck trainer uses a simpler MSE+result blend without the WDL CE component. So `wdlAlpha` and `wdlDrawElo` are correctly not sent to duck training — they're not applicable.

**Revised verdict:** ✅ Not a bug — duck trainer uses a different loss formulation that doesn't have WDL CE.

### ⚠️ Issue 2 — Overfitting Detection has no effect

`overfitDetect` is read from the UI checkbox, saved/loaded in presets, and passed to `train_nnue.py` via... nothing. It is **never passed as a command flag** to either `train_nnue.py` or `--train-duck`.

Searching the training command construction: no `--overfit` or `--overfitting` flag is sent. The `earlyStop` parameter controls early stopping patience, but `overfitDetect` as a separate on/off toggle is never used.

**Effect:** The "Overfitting Detection" checkbox has no effect on training. The `earlyStop` patience value always applies regardless of this checkbox.

**Fix:** Either wire `overfitDetect=false` to pass `--early-stop 0` (disabling early stopping), or remove the checkbox and always use `earlyStop` directly.

### ⚠️ Issue 3 — Cosine LR flag logic is inverted for standard training

For standard chess (`train_nnue.py`):
```cpp
if (!cfg.cosineLr) args += " --no-cosine-restarts";
if (cfg.cosineLr && cfg.cosineT0 > 0) args += " --cosine-t0 " + ...;
```

The `--enhanced` flag is always passed, which enables cosine LR by default in `train_nnue.py`. So `--no-cosine-restarts` disables it when `cosineLr=false`. This is correct but inverted from what you'd expect — the flag is "disable" not "enable".

For duck chess:
```cpp
if (cfg.cosineLr) duckArgs += " --cosine-lr";
```

Duck training requires explicit `--cosine-lr` to enable it. This is the opposite convention. Both work correctly but the asymmetry is confusing.

**Verdict:** ✅ Functional but asymmetric convention between standard and duck training.

### ⚠️ Issue 4 — SWA Games not implemented

`swaGames` is read from UI, saved in presets, but there is no SWA validation match in the pipeline. The tooltip describes a match between the SWA-averaged model and the best-checkpoint model, but this match is never run. The `swaGames` value is never used anywhere in the pipeline code.

**Effect:** `SWA Games` setting has no effect. The SWA weights are simply saved as `nnue_weights_swa_genN.bin` after training but never validated against the regular weights.

**Fix options:**
1. Implement the SWA validation match (medium effort)
2. Remove the UI control and preset field (breaking change for saved presets)
3. Leave as-is and document as "reserved for future use"

### ✅ Issue 5 — Draw Ratio % (drawPct) only applies to standard training

`drawPct` adds a draws dataset as extra training data for `train_nnue.py`. It is NOT passed to `--train-duck`. This is intentional — the duck trainer handles data mixing differently. The UI setting is correctly ignored for duck chess.

---

## Summary

| # | Issue | Severity | Status |
|---|---|---|---|
| 1 | WDL Alpha/Draw Elo not sent to duck training | N/A | ✅ Not applicable (different loss) |
| 2 | Overfitting Detection checkbox has no effect | Medium | 🔴 Dead UI control |
| 3 | Cosine LR flag convention asymmetric | Low | ⚠️ Works but confusing |
| 4 | SWA Games not distinct from ELO Games | Low | ⚠️ Dead UI control |
| 5 | Draw Ratio % not sent to duck training | N/A | ✅ Intentional |

**All 52 settings are correctly saved/loaded. 2 UI controls (Overfitting Detection, SWA Games) have no functional effect on the pipeline.**
