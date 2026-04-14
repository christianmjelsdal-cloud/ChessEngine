#pragma once
// ============================================================================
//  GameLogic.h — Pure, reusable game-rule functions (no GUI dependency)
//
//  Extracted from VisualGame.cpp so the same logic can be used by the
//  visual GUI, UCI mode, self-play generator, and tests.
//
//  All functions are pure or nearly-pure: they read the Board/MoveList but
//  never mutate them. No SFML headers, no threads, no I/O.
// ============================================================================

#include "Board.h"
#include "MoveGen.h"
#include "Types.h"
#include <vector>
#include <cstdint>
#include <string>

namespace GameLogic {

// ── Result classification ────────────────────────────────────────────────────

enum class GameResult {
    Ongoing,              // game is still in progress
    CheckmateWhiteWins,   // Black is checkmated
    CheckmateBlackWins,   // White is checkmated
    Stalemate,            // side-to-move has no legal moves and is not in check
    DrawByRepetition,     // threefold repetition detected
    DrawBy50Move,         // 50-move rule (halfMoveClock >= 100)
    DrawByInsufficientMaterial,
#ifdef DUCK_CHESS
    DuckNoMovesWhiteWins, // Black has no legal moves (duck chess — no stalemate)
    DuckNoMovesBlackWins, // White has no legal moves (duck chess)
    DuckKingCapturedWhiteWins,
    DuckKingCapturedBlackWins,
#endif
};

/// One-stop check: classify the current position's result.
/// @param board       Current board state (after the last move was applied).
/// @param legalMoves  Pre-computed legal moves for the side to move.
/// @param posHashes   Position history (Zobrist hashes) for repetition detection.
/// @param isDuckChess Whether duck chess rules apply.
GameResult classify(const Board& board,
                    const MoveList& legalMoves,
                    const std::vector<uint64_t>& posHashes,
                    bool isDuckChess = false);

// ── Individual rule checks ───────────────────────────────────────────────────

/// Is the side to move in checkmate?
bool isCheckmate(const Board& board, const MoveList& legalMoves);

/// Is the side to move in stalemate?
bool isStalemate(const Board& board, const MoveList& legalMoves);

/// Is the position a draw by the 50-move rule?
bool isDraw50Move(const Board& board);

/// Is the position a draw by threefold repetition?
/// @param posHashes  Vector of all Zobrist hashes seen so far (including current).
bool isThreefoldRepetition(const std::vector<uint64_t>& posHashes, int halfMoveClock);

/// Is the position drawn due to insufficient mating material?
/// Detects: K vs K, K+B vs K, K+N vs K, K+B vs K+B (same-coloured bishops).
bool isInsufficientMaterial(const Board& board);

// ── Status string (for UI / logging) ────────────────────────────────────────

/// Return a human-readable status string for the current position.
std::string statusString(const Board& board,
                         const MoveList& legalMoves,
                         GameResult result);

}  // namespace GameLogic
