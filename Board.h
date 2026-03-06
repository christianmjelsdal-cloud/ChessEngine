#pragma once
#include "Types.h"
#include <string>

class Board {
public:
    // The 8x8 grid of pieces
    Piece squares[8][8];

    // Game state
    Color turn = Color::White;
    bool castlingRights[2][2] = { {true, true}, {true, true} };
    // castlingRights[Color][0] = kingside, [1] = queenside

    Square enPassantTarget = { -1, -1 }; // invalid by default

    int halfMoveClock = 0;  // for 50-move rule
    int fullMoveNumber = 1;

    // === Duck Chess ===
    bool isDuckChess = false;
    Square duckSquare = { -1, -1 }; // current duck position ({-1,-1} = not yet placed)

    // Constructor
    Board();

    // Setup
    void setStartingPosition();
    void clearBoard();

    // Access
    Piece getPiece(Square sq) const;
    void setPiece(Square sq, Piece piece);

    // Move
    void applyMove(const Move& move);

    // Duck chess: place (or move) the duck to a new square
    void placeDuck(Square sq);

    // Utility
    void printBoard() const;
    bool isSquareAttacked(Square sq, Color byColor) const;

    // Check if a square is blocked by the duck (can't move to or through)
    bool isDuckSquare(Square sq) const;
    bool isDuckSquare(int rank, int col) const;
};
