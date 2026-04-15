# Analysis Window — Feature Roadmap

**Last updated:** April 14, 2026

---

## Overview

The right-side blank panel in ChessEngine.exe is reserved for a full analysis window.
The goal is a Lichess/ChessBase-style analysis experience integrated directly into the engine GUI.

---

## Feature List

### 1. Position Navigation ⬅ START HERE
Navigate back and forth through the game move by move.

**Controls:**
- `←` / `→` arrow keys — step one move back/forward
- `Ctrl+←` / `Ctrl+→` — jump to start/end of game
- Scroll wheel over board — step through moves
- Click any move in the move list — jump directly to that position

**Two modes:**
- **Live mode** — engine plays/thinks on the current game position (current behavior)
- **Analysis mode** — triggered by navigating back; engine analyzes the viewed position; indicator shows "Analysis — move 14"

**Data structure:**
```cpp
struct HistoryEntry {
    Board       board;       // full board snapshot at this position
    Move        movePlayed;  // move that leads to the next position
    std::string moveAlg;     // algebraic notation ("Nf3", "O-O", "exd5")
    int         evalBefore;  // engine eval at this position
    int         evalAfter;   // eval after the move was played
};
std::vector<HistoryEntry> gameHistory_;
int                       viewIdx_ = 0;  // currently viewed position index
```

**Implementation order:**
1. Add `HistoryEntry` vector + `viewIdx_` — store snapshots as moves are played
2. Wire `←`/`→` keys + scroll wheel to change `viewIdx_` and re-render historical board
3. Draw move list in right panel with current position highlighted
4. Add engine analysis of the viewed position (re-run `getBestMove` on snapshot)
5. Add centipawn loss annotation once evals are available for all positions

---

### 2. Multi-line PV Display
Show top 3–5 candidate moves with eval, depth, and move sequence.

- Engine already supports `multiPV` mode
- Display in right panel: `1. Nf3 +0.4  2. e4 +0.3  3. d4 +0.2`
- Each line shows: rank, best move, eval, continuation moves

---

### 3. Eval Graph
Plot centipawn evaluation over the course of the game.

- X-axis: move number
- Y-axis: centipawn eval (white positive, black negative)
- Highlights where mistakes/blunders occurred
- Click a point on the graph to jump to that position
- Similar infrastructure to TrainingRunner's loss graph

---

### 4. Centipawn Loss Per Move
After a game, annotate each move with how many centipawns were lost vs engine best.

**Classification thresholds (Lichess-style):**
| Label | CP Loss |
|-------|---------|
| Best | 0–5 |
| Good | 5–20 |
| Inaccuracy | 20–50 |
| Mistake | 50–100 |
| Blunder | 100+ |

Color-code moves in the move list accordingly.

---

### 5. Best Move + Alternative Arrows
Extend existing PV arrow drawing to show top 3 moves simultaneously.

- First choice: bright blue arrow
- Second choice: dimmer blue
- Third choice: dimmer still
- Opponent's best response: red arrow

---

### 6. Live Search Stats Display
Show real-time engine stats during analysis:

```
depth 18 | 2.4M nodes | 850K nps | 1.2s
```

Already available from engine — just needs rendering in the panel.

---

### 7. Algebraic Notation
Convert moves to standard algebraic notation for the move list.

- "Nf3", "exd5", "O-O", "Qxf7+", "e8=Q#"
- Requires disambiguation logic (two knights can go to f3 → "Ndf3")
- Check/checkmate suffix (+/#)
- Can start with coordinate notation ("e2e4") and upgrade

---

### 8. FEN Input/Output
Paste a FEN string to jump to any position for analysis.

- Text input field in the panel
- `Ctrl+V` to paste FEN, Enter to load
- Also copy current position FEN to clipboard

---

### 9. Opening Name Display
Match current position against ECO database and show opening name.

- Requires an ECO `.tsv` or `.json` file (freely available)
- Display: "Sicilian Defense, Najdorf Variation (B90)"
- Only relevant for first ~15 moves

---

### 10. Tablebase Display
For endgames with ≤5 pieces, show Syzygy result in the panel.

- Already have Syzygy integrated in the engine
- Display: "White wins in 14 moves (DTM)" or "Draw"
- Activate automatically when piece count drops to ≤5

---

### 11. PGN Export
Save the game with engine annotations to a `.pgn` file.

- Standard format, opens in any chess software
- Include eval annotations: `{ +0.34/18 }` after each move
- Blunder annotations: `$4` (blunder), `$2` (mistake), `$6` (inaccuracy)

---

### 12. Threat Detection
Highlight what the opponent is threatening if you don't respond.

- Run a quick search on the opponent's position
- Show the threat as a red arrow or text: "Threatens Qxf7#"

---

## Implementation Priority

| # | Feature | Effort | Value |
|---|---------|--------|-------|
| 1 | Position navigation + move list | Medium | ⭐⭐⭐⭐⭐ |
| 2 | Multi-line PV display | Low | ⭐⭐⭐⭐⭐ |
| 3 | Eval graph | Medium | ⭐⭐⭐⭐ |
| 4 | Centipawn loss | Low (needs #1) | ⭐⭐⭐⭐ |
| 5 | Alternative arrows | Low | ⭐⭐⭐⭐ |
| 6 | Live stats display | Very low | ⭐⭐⭐ |
| 7 | Algebraic notation | Medium | ⭐⭐⭐⭐ |
| 8 | FEN input/output | Low | ⭐⭐⭐ |
| 9 | Opening names | Low | ⭐⭐ |
| 10 | Tablebase display | Very low | ⭐⭐⭐ |
| 11 | PGN export | Medium | ⭐⭐⭐ |
| 12 | Threat detection | High | ⭐⭐ |

---

## Next Step

Implement **Feature 1: Position Navigation** — steps 1–3 (history storage, keyboard/scroll navigation, move list rendering).

### 13. Line Continuation (PV Expansion) ✅ Done
Lichess-style PV display with eval pill, move continuation, and expand/collapse.

- Each analysis line shows: `[+0.22]  c5 Nf3 Nc6 d4 cxd4 Nxd4  ^`
- Eval pill: rounded rectangle with score (green = white ahead, red = black ahead)
- First 6 moves shown by default; `^` expands to show full continuation
- `v` collapses back to 6 moves
- Top 3 lines shown, separated by dividers
- PV captured per root move during search (stored in `RootMove.pv`)

| # | Feature | Status |
|---|---------|--------|
| 1 | Position navigation + move list | ✅ Done |
| 2 | Multi-line PV display | ✅ Done (Lichess-style: eval pill + moves + expand toggle) |
| 3 | Eval graph | ✅ Done (sparkline) |
| 4 | Centipawn loss | ✅ Done (color-coded) |
| 5 | Alternative arrows | ✅ Done (green arrow on Z) |
| 6 | Live stats display | ✅ Done |
| 7 | Algebraic notation | ✅ Done |
| 8 | FEN input/output | ✅ Done (Ctrl+C/V) |
| 9 | Opening names | ✅ Done (Lichess ECO database, 3695 openings) |
| 10 | Tablebase display | ✅ Done (Syzygy WDL) |
| 11 | PGN export | ✅ Done (Ctrl+S) |
| 12 | Threat detection | ✅ Done (red arrow = best opponent capture) |

## Status

| # | Feature | Status |
|---|---------|--------|
| 1 | Position navigation + move list | ✅ Done |
| 2 | Multi-line PV display | ✅ Done (Lichess-style eval pill + continuation + expand) |
| 3 | Eval graph | ✅ Done (filled areas, eval label on marker) |
| 4 | Centipawn loss | ✅ Done (color-coded moves) |
| 5 | Alternative arrows | ✅ Done (green = best, red = threat) |
| 6 | Live stats display | ✅ Done |
| 7 | Algebraic notation | ✅ Done |
| 8 | FEN input/output | ✅ Done (Ctrl+C/V) |
| 9 | Opening names | ✅ Done (Lichess ECO, 3695 openings) |
| 10 | Tablebase display | ✅ Done (Syzygy WDL) |
| 11 | PGN export | ✅ Done (Ctrl+S) |
| 12 | Threat detection | ✅ Done (red arrow = best opponent capture) |
| 13 | Line continuation | ✅ Done (PV expansion with ^ / v toggle) |
