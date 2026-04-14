#pragma once
#include "Board.h"
#include <string>

namespace Syzygy {

/// Initialize tablebases from the given directory path.
/// Returns true if at least one TB file was found.
bool init(const std::string& path);

/// Cleanup tablebase memory.
void free();

/// Returns the largest piece count available (0 if not initialized).
int maxPieces();

/// Count total pieces on the board.
int pieceCount(const Board& board);

/// WDL probe for use in search. Returns score from side-to-move perspective.
/// Returns false if probe failed (position not in TB, too many pieces, castling rights, etc.)
/// score: positive = win, 0 = draw, negative = loss. Uses large values (±TB_WIN_SCORE).
bool probeWDL(const Board& board, int& score);

/// Root probe with DTZ. Returns the best move according to DTZ.
/// Also returns WDL score. Returns false if probe failed.
bool probeRoot(const Board& board, Move& bestMove, int& score);

/// Score constants
constexpr int TB_WIN_SCORE = 20000;   // Must be < MATE but >> any eval
constexpr int TB_CURSED_WIN = 50;     // Cursed win (win but 50-move rule draws)
constexpr int TB_BLESSED_LOSS = -50;  // Blessed loss

} // namespace Syzygy
