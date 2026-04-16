# Chess Engine

A C++17 chess engine featuring NNUE evaluation, Lazy SMP parallel search, and
an optional SFML-based graphical interface.

## Features

| Feature | Description |
|---------|-------------|
| **Alpha-beta search** | Iterative deepening + PVS with aspiration windows |
| **NNUE evaluation** | 768→256→1 half-KP network with AVX2/FMA/SSE4.1 SIMD |
| **Lazy SMP** | Multi-threaded search (configurable thread count) |
| **Move ordering** | MVV-LVA, killer moves, history heuristic, countermove history, SEE |
| **Reductions/extensions** | LMR, null-move pruning, singular extensions, check extensions |
| **Transposition table** | Lockless Zobrist-indexed TT with generational aging |
| **UCI protocol** | Standard UCI interface for GUI integration |
| **Duck Chess** | Optional variant (compile with `-DDUCK_CHESS`) |
| **Self-play training** | Built-in self-play data generator and NNUE trainer |

## Requirements

- **Compiler**: C++17 (MSVC 2022, GCC 12+, Clang 15+)
- **CPU**: x86-64 with AVX2 + FMA (Haswell or newer)
- **SFML 3.x**: Required only for the GUI (Visual Studio project)
- **CMake 3.14+**: For command-line / CI builds

## Building

### Visual Studio (Windows — GUI)

1. Install [SFML 3.x](https://www.sfml-dev.org/) via vcpkg or manually
2. Open `ChessEngine.sln`
3. Build in Release mode (x64)

### CMake (cross-platform — command-line / tests)

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
```

This produces:
- `test_exe` — Perft test suite (move generation validation)
- `smoketest_exe` — Engine smoke tests (search + evaluation sanity)

> **Note**: The CMake build does not include the SFML GUI (`VisualGame`).
> Use the Visual Studio project for the full graphical application.

## Running Tests

### Perft Tests (move generation correctness)

```bash
./build/test_exe
```

Runs 26 perft tests across 6 standard positions (depths 1–5), verifying
that the move generator produces the exact published node counts. Also
checks bitboard↔mailbox consistency at every node.

Expected output: `26/26 passed` in under 1 second.

### Smoke Tests (search engine behaviour)

```bash
./build/smoketest_exe
```

Runs 10 rapid-search tests covering:
- Mate-in-1 detection
- Basic tactics (hanging piece capture)
- Endgame play (K+R vs K)
- Promotion handling
- Castling and en passant positions
- Stalemate avoidance

All tests use the handcrafted evaluator (no NNUE weights required).

### UCI Mode

```bash
./ChessEngine --uci
```

Or connect via any UCI-compatible chess GUI (Arena, Cute Chess, etc.).

## Project Structure

| File | Description |
|------|-------------|
| `Types.h` | Core types (Piece, Square, Move, MoveList) + board geometry constants |
| `Board.h/cpp` | Board representation: mailbox + bitboards, make/unmake, FEN |
| `Bitboard.h/cpp` | Bitboard attack tables and utility functions |
| `MoveGen.h/cpp` | Legal move generation (pins, checks, en passant, castling, promotions) |
| `Engine.h/cpp` | Alpha-beta search engine with all search enhancements |
| `NNUE.h/cpp` | NNUE network: inference with AVX2 SIMD + quantised accumulators |
| `GameLogic.h/cpp` | Pure game rules: checkmate, stalemate, draw detection |
| `UCI.h/cpp` | UCI protocol handler |
| `VisualGame.h/cpp` | SFML graphical interface (requires SFML 3.x) |
| `Test.h/cpp` | Perft test framework |
| `SmokeTest.cpp` | Engine behavioural smoke tests |
| `NNUETrainer.h/cpp` | NNUE training pipeline |
| `SelfPlayGen.h/cpp` | Self-play game generator for training data |
| `CMakeLists.txt` | Cross-platform build for tests (no SFML required) |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                      UCI / GUI                       │
├─────────────────────────────────────────────────────┤
│                  Engine (search)                     │
│   ┌────────┐  ┌────────┐  ┌──────────┐  ┌───────┐  │
│   │  PVS   │  │  LMR   │  │ Null-Move│  │  SEE  │  │
│   └────────┘  └────────┘  └──────────┘  └───────┘  │
├───────────────────┬─────────────────────────────────┤
│   NNUE eval       │   Handcrafted eval (fallback)   │
│  768→256→1        │   Material + PST + mobility     │
├───────────────────┴─────────────────────────────────┤
│            Board + MoveGen + Bitboards               │
│     mailbox[8][8] ↔ bitboards (synced)              │
└─────────────────────────────────────────────────────┘
```

## Duck Chess Variant

Compile with `-DDUCK_CHESS` to enable Duck Chess mode:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-DDUCK_CHESS"
```

In Duck Chess, after making a normal move, the player places a "duck" on
any empty square. The duck blocks movement for both sides. The win
condition is king capture (not checkmate), and stalemate is a loss.

## License

See repository for license details.
