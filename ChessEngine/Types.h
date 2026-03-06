#pragma once
#include <cstdint>

enum class PieceType : uint8_t {
    None, Pawn, Knight, Bishop, Rook, Queen, King, Duck
};

enum class Color : uint8_t {
    White, Black
};

struct Piece {
    PieceType type = PieceType::None;
    Color color = Color::White;
    bool isNone() const { return type == PieceType::None; }
    bool isDuck() const { return type == PieceType::Duck; }
};

struct Square {
    int rank; // 0-7 (row, 0 = white's back rank)
    int col;  // 0-7 (column, 0 = a-file)
    bool isValid() const { return rank >= 0 && rank < 8 && col >= 0 && col < 8; }
    bool operator==(const Square& o) const { return rank == o.rank && col == o.col; }
    bool operator!=(const Square& o) const { return !(*this == o); }
};

struct Move {
    Square from = { 0, 0 };
    Square to = { 0, 0 };
    PieceType promotion = PieceType::None; // for pawn promotion
    Square duckTo = { -1, -1 };            // duck placement (duck chess only)
    // NOTE: operator== intentionally ignores duckTo so the engine's
    // killer/hash/history logic works unchanged. The duck placement
    // is handled as a separate search phase inside the engine.
    bool operator==(const Move& o) const {
        return from == o.from && to == o.to && promotion == o.promotion;
    }
    bool operator!=(const Move& o) const { return !(*this == o); }
};
