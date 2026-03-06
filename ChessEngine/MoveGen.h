#pragma once
#include "Board.h"
#include <vector>

class MoveGen {
public:
    // Returns all legal moves for the current side to move
    static std::vector<Move> getLegalMoves(const Board& board);

    // Returns all pseudo-legal moves (ignores if king is left in check)
    static std::vector<Move> getPseudoLegalMoves(const Board& board);

    static bool isInCheck(const Board& board, Color color);

    // === Duck Chess ===
    // Returns chess moves in duck chess mode (no check filtering, duck blocks)
    static std::vector<Move> getDuckChessMoves(const Board& board);

    // Returns all valid duck placement squares after a chess move is applied
    static std::vector<Square> getDuckPlacements(const Board& board);

    // Check if a king was captured (duck chess win condition)
    static bool isKingCaptured(const Board& board, Color color);

private:
    static void generatePawnMoves(const Board& board, Square sq, std::vector<Move>& moves);
    static void generateKnightMoves(const Board& board, Square sq, std::vector<Move>& moves);
    static void generateBishopMoves(const Board& board, Square sq, std::vector<Move>& moves);
    static void generateRookMoves(const Board& board, Square sq, std::vector<Move>& moves);
    static void generateQueenMoves(const Board& board, Square sq, std::vector<Move>& moves);
    static void generateKingMoves(const Board& board, Square sq, std::vector<Move>& moves);
};
