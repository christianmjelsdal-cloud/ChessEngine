# TrainingRunner.cpp — Improvement Recommendations

A thorough review of the 5,039-line NNUE Training Runner GUI. These recommendations are organized from highest-impact to nice-to-haves.

---

## 🔴 Bugs (Fix These First)

### 1. Control ID Collision: `ID_CHK_MUTE_SOUNDS` and `ID_EDIT_MIXDEPTH_PCT`
Both are set to **1054**. This means clicking "Mute Sounds" and changing "Mixed Depth %" will interfere with each other — Windows can't distinguish which control sent `WM_COMMAND`.

```cpp
ID_CHK_MUTE_SOUNDS   = 1054,  // ← COLLISION
ID_EDIT_MIXDEPTH_PCT = 1054,  // ← COLLISION
```

**Fix:** Change `ID_EDIT_MIXDEPTH_PCT` to 1060 (or any unused value) and renumber the following IDs.

### 2. Embedded `elo_match.py` Has a Syntax Error
Line ~2482 in the embedded Python script:
```python
subprocess.check_call([sys.executable, "-m", ""pip", "install", ...
```
There's an extra `"` before `pip`. This will crash the first time it tries to install `python-chess`.

**Fix:** Change `""pip"` to `"pip"`.

### 3. `genStatsPath()` Calls `ReadConfig()` — Dangerous Recursion Risk
The comment says "must not be called from within ReadConfig()", but there's no compile-time or runtime guard. If someone adds a genStatsPath() call inside ReadConfig in the future, it'll infinitely recurse and stack overflow.

**Fix:** Have `genStatsPath()` accept the data dir as a parameter instead of calling ReadConfig().

---

## 🟠 Architecture Improvements (High Impact)

### 4. Split the File — 5,039 Lines Is Too Large
This single file handles UI layout, GDI+ graph rendering, subprocess management, config serialization, benchmarking, pipeline orchestration, embedded Python scripts, WMI temperature queries, and more. This makes it very hard to maintain.

**Suggested split:**
| File | Contents |
|------|----------|
| `TrainingRunner.cpp` | `main()`, window creation, message loop |
| `Pipeline.cpp/h` | `PipelineThread`, `SelfPlay`, `Training`, `EloVal`, `SwaBestOfTwo` |
| `Graph.cpp/h` | `DrawGraph`, `GraphProc`, all graph rendering |
| `Config.cpp/h` | `Config`, `Preset`, serialization, preset management |
| `Benchmark.cpp/h` | `BenchmarkThread`, `RecalibrateTimePresets`, CPU temp |
| `Process.cpp/h` | `RunProc`, process tree management, suspend/resume |
| `UI.cpp/h` | `BuildConfigPane`, helper controls, tooltip system |

### 5. Unify `Config` and `Preset` Structs
`Config` and `Preset` have ~35 identical fields with duplicated defaults. Every time you add a parameter, you must update both structs, `SerializePreset`, `DeserializePreset`, `ReadConfig`, `ApplyPreset`, and `SavePresetAs`.

**Fix:** Make `Preset` inherit from or contain a `Config`:
```cpp
struct Preset {
    std::string name;
    bool isBuiltin = false;
    Config cfg;  // all training parameters in one place
};
```

### 6. Reduce Global State
There are **50+ global variables**. This makes the code hard to reason about, test, and refactor. Consider grouping related globals into structs:

```cpp
struct UIState {
    HWND hStart, hStop, hPause, hSkip, hPreset, ...;
};
struct GraphState {
    int hoverIdx = -1;
    POINT mousePt = {-1, -1};
    int hoverPanel = -1;
    GraphPanelBounds panelBounds[4];
    bool showLoss = true, showAcc = true, showLR = true, showPhase = true;
};
```

---

## 🟡 Performance & Robustness

### 7. Graph Redraws Every 500ms Regardless of Change
The timer invalidates the graph every tick. The "cache" only checks `pts.size()` — so if a point's value changes (in-place updates), the cache serves stale data. And when nothing changes at all, you're still doing a full GDI+ render.

**Fix:** Use a dirty flag set by `pushPt()` and phase changes. Only invalidate when actually dirty.

### 8. Hardcoded Python Version `py -3.10`
Training and benchmark commands use `py -3.10`. If the user has Python 3.11 or 3.12, this fails silently or errors out.

**Fix:** Make the Python command configurable (add to Config), or probe for available versions:
```cpp
// Try py -3.10, fall back to py -3, fall back to python
```

### 9. Thread Detach on Shutdown Is a Data Race
If the worker thread doesn't exit within 5 seconds, it's detached. But it still references globals (`g_st`, `g_cfg`, file paths) which are being destroyed. This can cause crashes on exit.

**Fix:** Use `TerminateProcess` on the active child process (already done via `SuspendOrTerminateActive`), then give the worker more time. Or use a shared_ptr to state so it stays alive.

### 10. No Validation of Config Values Before Pipeline Start
Users can enter nonsensical values: negative games, 0 batch size, learning rate of 999, etc. The pipeline will start and fail confusingly.

**Fix:** Add a `ValidateConfig()` function that checks ranges before `PipelineThread` launches:
```cpp
if (cfg.batchSize <= 0 || cfg.batchSize > 65536) { error("Batch size must be 1-65536"); return; }
if (cfg.lr <= 0 || cfg.lr > 1.0) { error("Learning rate must be 0-1"); return; }
// etc.
```

### 11. COM Not Initialized on UI Thread
`GetCpuTempCelsius()` uses COM (WMI) and `CoCreateInstance`. It's only safe to call from threads that have called `CoInitialize`. The benchmark thread does this, but if this function were ever called from the UI thread, it would fail or crash.

**Fix:** Initialize COM on the UI thread in `main()` as well, or add a guard inside `GetCpuTempCelsius()`.

---

## 🟢 Code Quality

### 12. Duplicated `fmtDur` Lambda
The duration-formatting lambda is defined twice: once in the timer handler and once in the "pipeline done" branch. Extract it to a standalone function:
```cpp
static std::wstring fmtDuration(long long totalSec) { ... }
```

### 13. Embedded Python Scripts Should Be External Files
Three complete Python scripts (elo_match.py, swa_match.py, and the UCI engine wrapper) are embedded as C++ raw string literals. This means:
- No syntax highlighting, no linting, no IDE support
- Hard to test independently
- Bloats the C++ file enormously (~400 lines of Python in C++)

**Fix:** Ship them as separate `.py` files in the assets folder. The current `FIX 6.30` clobber guard already moves in this direction.

### 14. Magic Numbers Throughout
Examples:
- `500` (ms timer interval) — use a named constant
- `800` (MAX_LOG) — already named, good
- `0.55` (self-play budget fraction) — should be a named constant
- `3.0` (depth extrapolation factor) — should be documented
- `30000` (command line length warning) — use `30000` as a named constant

### 15. Tooltip Text Is Enormously Long
Some tooltips are 800+ characters. While thorough, they make the code very hard to read. Consider moving tooltip strings to a separate resource file or string table.

### 16. `stripToken` Lambda Is Complex and Fragile
The ANSI-stripping + token extraction lambda inside `Training()` is 40+ lines. It handles edge cases for `[K`, ANSI escapes, pipe-delimited fields, etc. This deserves to be a proper function with unit tests.

### 17. No Logging to File
All logging goes to the in-memory deque and UI listbox. If the app crashes, all diagnostic info is lost.

**Fix:** Optionally write log lines to a timestamped file in the assets directory (e.g., `training_log_2026-03-26.txt`).

---

## 🔵 Feature Ideas

### 18. Export Training History
Users can see the graph but can't export the data. Adding a "Save CSV" button for the training points would let users analyze runs in Excel/matplotlib.

### 19. Multi-Run Comparison
No way to overlay results from previous runs on the graph. Even just loading a previous `gen_stats.csv` as a reference line would be valuable.

### 20. System Tray Integration
Long runs could show a tray icon with progress. The `PlayMp3` notifications are great but only work when the window is in focus.

### 21. Auto-Save Config on Start
If the app crashes mid-run, the config is lost unless it was saved as a preset. Auto-saving the running config to a `last_run.cfg` file would help.

---

## Summary

| Priority | Count | Key Items |
|----------|-------|-----------|
| 🔴 Bugs | 3 | ID collision (1054), Python syntax error, recursion risk |
| 🟠 Architecture | 3 | Split file, unify Config/Preset, reduce globals |
| 🟡 Robustness | 5 | Config validation, Python version, graph perf, COM init, shutdown |
| 🟢 Code Quality | 6 | Dedupe fmtDur, externalize Python, magic numbers, logging |
| 🔵 Features | 4 | CSV export, multi-run comparison, tray icon, auto-save |

The three bugs should be fixed immediately — especially the **ID collision** (causes real user-facing misbehavior) and the **Python syntax error** (will crash on first run if `python-chess` isn't pre-installed).
