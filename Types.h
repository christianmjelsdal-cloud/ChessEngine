#pragma once
#include <cassert>
#include <cstdint>
#include <type_traits>

// ── Board geometry constants ─────────────────────────────────────────────────
// Replace magic numbers (0, 7, 4, 2…) in Board.cpp / MoveGen.cpp with these
// named constants for readability and safety.

constexpr int BOARD_SIZE = 8;

// File indices (columns)
constexpr int FILE_A = 0;
constexpr int FILE_B = 1;
constexpr int FILE_C = 2;
constexpr int FILE_D = 3;
constexpr int FILE_E = 4;  // king start file
constexpr int FILE_F = 5;
constexpr int FILE_G = 6;
constexpr int FILE_H = 7;

// Rank indices (rows — rank 1 = index 0 = White's back rank)
constexpr int RANK_1 = 0;
constexpr int RANK_2 = 1;
constexpr int RANK_3 = 2;
constexpr int RANK_4 = 3;
constexpr int RANK_5 = 4;
constexpr int RANK_6 = 5;
constexpr int RANK_7 = 6;
constexpr int RANK_8 = 7;

// Piece starting positions
constexpr int KING_START_FILE        = FILE_E;  // e-file
constexpr int ROOK_KINGSIDE_FILE     = FILE_H;  // h-file
constexpr int ROOK_QUEENSIDE_FILE    = FILE_A;  // a-file

// Back rank for each colour (index into board[rank][col])
constexpr int WHITE_BACK_RANK = RANK_1;  // 0
constexpr int BLACK_BACK_RANK = RANK_8;  // 7

// Pawn start ranks (for double-push detection)
constexpr int WHITE_PAWN_START_RANK = RANK_2;  // 1
constexpr int BLACK_PAWN_START_RANK = RANK_7;  // 6

// Move deltas
constexpr int CASTLING_KING_DELTA    = 2;  // king moves 2 squares when castling
constexpr int DOUBLE_PAWN_DELTA      = 2;  // pawn moves 2 squares on first move

// Castling rights array indices
constexpr int CASTLE_KINGSIDE  = 0;
constexpr int CASTLE_QUEENSIDE = 1;

// Draw thresholds
constexpr int FIFTY_MOVE_CLOCK_LIMIT = 100;  // halfMoveClock >= 100 = draw
constexpr int THREEFOLD_REP_COUNT    = 3;    // 3 repetitions = draw

// ── Piece types ──────────────────────────────────────────────────────────────

enum class PieceType : uint8_t {
    None, Pawn, Knight, Bishop, Rook, Queen, King
#ifdef DUCK_CHESS
    , Duck
#endif
};

enum class Color : uint8_t {
    White, Black
};

struct Piece {
    PieceType type = PieceType::None;
    Color color = Color::White;
    bool isNone() const { return type == PieceType::None; }
    bool operator==(const Piece& o) const { return type == o.type && color == o.color; }
    bool operator!=(const Piece& o) const { return !(*this == o); }
#ifdef DUCK_CHESS
    bool isDuck() const { return type == PieceType::Duck; }
#else
    bool isDuck() const { return false; }
#endif
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
#ifdef DUCK_CHESS
    Square duckTo = { -1, -1 };            // duck placement (duck chess only)
#endif
    // NOTE: operator== intentionally ignores duckTo so the engine's
    // killer/hash/history logic works unchanged. The duck placement
    // is handled as a separate search phase inside the engine.
    bool operator==(const Move& o) const {
        return from == o.from && to == o.to && promotion == o.promotion;
    }
    bool operator!=(const Move& o) const { return !(*this == o); }
};

// AUDIT FIX BUG-4: Compile-time guard — Move must remain trivially copyable.
// std::fill and memset on Move arrays are safe only while this holds.
static_assert(std::is_trivially_copyable_v<Move>, "Move must be trivially copyable for std::fill/memset safety");
static_assert(std::is_trivially_copyable_v<Square>, "Square must be trivially copyable");

// TIER-2 FIX #7: Pack/unpack Move into 16 bits for TT storage
inline uint16_t packMove(const Move& m) {
    int from = m.from.rank * 8 + m.from.col;
    int to   = m.to.rank * 8 + m.to.col;
    assert(from >= 0 && from < 64 && "packMove: 'from' square out of range");
    assert(to >= 0 && to < 64 && "packMove: 'to' square out of range");
    // Clamp to valid range in release builds to prevent silent bit corruption
    if (from < 0 || from >= 64 || to < 0 || to >= 64) return 0;
    return (uint16_t)((from << 10) | (to << 4) | ((int)m.promotion & 0xF));
}
inline Move unpackMove(uint16_t packed) {
    Move m;
    int from = (packed >> 10) & 63;
    int to   = (packed >> 4) & 63;
    m.from = { from / 8, from % 8 };
    m.to   = { to / 8, to % 8 };
    // FIX 8.6: Clamp promotion to valid PieceType range to prevent OOB from corrupt TT data
    int promRaw = packed & 0xF;
    m.promotion = (promRaw <= static_cast<int>(PieceType::King))
        ? static_cast<PieceType>(promRaw) : PieceType::None;
    return m;
}
inline bool isPackedMoveValid(uint16_t packed) {
    return packed != 0;
}

// §1.2: Stack-allocated move list for zero-heap-allocation search
struct MoveList {
    static constexpr int MAX_MOVES = 256;  // max legal moves in chess is 218
    Move moves[MAX_MOVES];
    int  count = 0;

    void add(const Move& m) {
        assert(count < MAX_MOVES && "MoveList overflow");
        if (count < MAX_MOVES) { moves[count++] = m; }
    }
    void push_back(const Move& m) { add(m); }  // compatibility alias
    void clear() { count = 0; }
    bool empty() const { return count == 0; }
    int  size() const { return count; }
    Move& operator[](int i) { assert(i >= 0 && i < count && "MoveList index out of bounds"); return moves[i]; }
    const Move& operator[](int i) const { assert(i >= 0 && i < count && "MoveList index out of bounds"); return moves[i]; }

    // Iterator support for range-for loops
    Move*       begin()       { return moves; }
    Move*       end()         { return moves + count; }
    const Move* begin() const { return moves; }
    const Move* end()   const { return moves + count; }
};

// Stack-allocated square list (e.g., duck placements) — mirrors MoveList pattern
struct SquareList {
    static constexpr int MAX_SQUARES = 64;
    Square squares[MAX_SQUARES] = {};
    int    count = 0;

    void add(const Square& s) {
        assert(count < MAX_SQUARES && "SquareList overflow");
        if (count < MAX_SQUARES) { squares[count++] = s; }
    }
    void push_back(const Square& s) { add(s); }
    void clear() { count = 0; }
    bool empty() const { return count == 0; }
    int  size() const { return count; }
    Square& operator[](int i) { assert(i >= 0 && i < count && "SquareList index out of bounds"); return squares[i]; }
    const Square& operator[](int i) const { assert(i >= 0 && i < count && "SquareList index out of bounds"); return squares[i]; }

    Square*       begin()       { return squares; }
    Square*       end()         { return squares + count; }
    const Square* begin() const { return squares; }
    const Square* end()   const { return squares + count; }
};

// NOTE: Board::UndoInfo (defined in Board.h) is the undo struct used everywhere.
// The legacy UndoInfo that was here has been removed.
