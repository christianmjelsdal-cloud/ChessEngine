#pragma once
// ============================================================================
//  Test.h — Perft (performance test) framework for move generation validation
//
//  Perft counts the total number of leaf nodes (positions) reachable at a
//  given depth from a given position, applying only legal moves. By comparing
//  against externally verified reference values, we validate that the board
//  representation, move generation, make/unmake, en passant, castling,
//  promotion, and check detection are all correct.
//
//  Usage:
//    BB::initBitboards();
//    Engine::initZobrist();
//    return runPerftSuite();   // 0 = all pass, 1 = some failed
// ============================================================================

#include "Board.h"
#include "MoveGen.h"
#include <cstdint>
#include <string>

/// Count leaf nodes at the given depth from the current position.
/// depth == 0 returns 1 (the current position is a leaf).
uint64_t perft(Board& board, int depth);

/// Run the full perft test suite. Returns 0 on success, 1 on any failure.
int runPerftSuite();

/// Validate that bitboard state matches mailbox state.
/// Returns true if consistent, false (and prints details) if not.
bool validateBoardConsistency(const Board& board, const std::string& context = "");
