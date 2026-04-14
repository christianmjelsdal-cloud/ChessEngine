#pragma once
#include "Board.h"
#include <vector>

class MoveGen {
public:
    // §1.2: Stack-allocated MoveList versions (primary — no heap allocation)
    // AUDIT FIX C-4: Accept Board& (not const) since we use makeMove/unmakeMove internally
    static void getLegalMoves(Board& board, MoveList& out);
    static void getLegalCaptures(Board& board, MoveList& out);
    static void getPseudoLegalMoves(const Board& board, MoveList& out);

    static bool isInCheck(const Board& board, Color color);

#ifdef DUCK_CHESS
    // === Duck Chess ===
    static void getDuckChessMoves(const Board& board, MoveList& out);
    static void getDuckPlacements(const Board& board, SquareList& out);
    static bool isKingCaptured(const Board& board, Color color);
#endif

private:
    static void generatePawnMoves(const Board& board, Square sq, MoveList& moves);
    static void generateKnightMoves(const Board& board, Square sq, MoveList& moves);
    static void generateBishopMoves(const Board& board, Square sq, MoveList& moves);
    static void generateRookMoves(const Board& board, Square sq, MoveList& moves);
    static void generateQueenMoves(const Board& board, Square sq, MoveList& moves);
    static void generateKingMoves(const Board& board, Square sq, MoveList& moves);
};
