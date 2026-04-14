#include "Syzygy.h"
// ── MSVC compatibility: popcount ────────────────────────────────────────────
#ifdef _MSC_VER
#  include <intrin.h>
#  pragma intrinsic(__popcnt64)
   static inline int popcount(unsigned long long x) { return (int)__popcnt64(x); }
#elif !defined(__GNUC__) && !defined(__clang__)
#  include <bit>
   static inline int popcount(unsigned long long x) { return std::popcount(x); }
#endif
// ────────────────────────────────────────────────────────────────────────────

#include "Bitboard.h"

// Fathom library — user must add tbprobe.h and tbprobe.c to their build
#ifdef HAS_SYZYGY
extern "C" {
#include "tbprobe.h"
}
#endif

namespace Syzygy {

static bool initialized_ = false;

bool init(const std::string& path) {
#ifdef HAS_SYZYGY
    if (path.empty()) return false;
    initialized_ = tb_init(path.c_str());
    return initialized_;
#else
    (void)path;
    return false;
#endif
}

void free() {
#ifdef HAS_SYZYGY
    if (initialized_) {
        tb_free();
        initialized_ = false;
    }
#endif
}

int maxPieces() {
#ifdef HAS_SYZYGY
    return initialized_ ? TB_LARGEST : 0;
#else
    return 0;
#endif
}

int pieceCount(const Board& board) {
    return popcount(board.occupiedBB);
}

// Helper: check if position is TB-eligible
static bool canProbe(const Board& board) {
#ifdef HAS_SYZYGY
    if (!initialized_) return false;
    // Too many pieces
    if (pieceCount(board) > (int)TB_LARGEST) return false;
    // Castling rights invalidate TB
    if (board.castlingRights[0][0] || board.castlingRights[0][1] ||
        board.castlingRights[1][0] || board.castlingRights[1][1])
        return false;
    return true;
#else
    (void)board;
    return false;
#endif
}

static unsigned epSquare(const Board& board) {
    if (board.enPassantTarget.rank < 0) return 0;
    return (unsigned)(board.enPassantTarget.rank * 8 + board.enPassantTarget.col);
}

bool probeWDL(const Board& board, int& score) {
#ifdef HAS_SYZYGY
    if (!canProbe(board)) return false;

    unsigned result = tb_probe_wdl(
        board.colorBB[0],  // white pieces
        board.colorBB[1],  // black pieces
        board.pieces(PieceType::King),
        board.pieces(PieceType::Queen),
        board.pieces(PieceType::Rook),
        board.pieces(PieceType::Bishop),
        board.pieces(PieceType::Knight),
        board.pieces(PieceType::Pawn),
        (unsigned)board.halfMoveClock, // rule50
        0u,                            // castling (filtered by canProbe)
        epSquare(board),
        board.turn == Color::White
    );

    if (result == TB_RESULT_FAILED) return false;

    switch (result) {
        case TB_WIN:          score = TB_WIN_SCORE;    break;
        case TB_CURSED_WIN:   score = TB_CURSED_WIN;   break;
        case TB_DRAW:         score = 0;               break;
        case TB_BLESSED_LOSS: score = TB_BLESSED_LOSS; break;
        case TB_LOSS:         score = -TB_WIN_SCORE;   break;
        default: return false;
    }
    return true;
#else
    (void)board; (void)score;
    return false;
#endif
}

bool probeRoot(const Board& board, Move& bestMove, int& score) {
#ifdef HAS_SYZYGY
    if (!canProbe(board)) return false;

    unsigned results[256];
    unsigned result = tb_probe_root(
        board.colorBB[0],
        board.colorBB[1],
        board.pieces(PieceType::King),
        board.pieces(PieceType::Queen),
        board.pieces(PieceType::Rook),
        board.pieces(PieceType::Bishop),
        board.pieces(PieceType::Knight),
        board.pieces(PieceType::Pawn),
        board.halfMoveClock,
        0u,                 // castling (filtered by canProbe)
        epSquare(board),
        board.turn == Color::White,
        results
    );

    if (result == TB_RESULT_FAILED) return false;

    // Extract move from result
    unsigned from = TB_GET_FROM(result);
    unsigned to   = TB_GET_TO(result);
    unsigned prom = TB_GET_PROMOTES(result);
    unsigned wdl  = TB_GET_WDL(result);

    bestMove.from = Square{(int)(from / 8), (int)(from % 8)};
    bestMove.to   = Square{(int)(to / 8), (int)(to % 8)};

    // Map Fathom promotion to our PieceType
    switch (prom) {
        case TB_PROMOTES_QUEEN:  bestMove.promotion = PieceType::Queen;  break;
        case TB_PROMOTES_ROOK:   bestMove.promotion = PieceType::Rook;   break;
        case TB_PROMOTES_BISHOP: bestMove.promotion = PieceType::Bishop; break;
        case TB_PROMOTES_KNIGHT: bestMove.promotion = PieceType::Knight; break;
        default:                 bestMove.promotion = PieceType::None;   break;
    }

    switch (wdl) {
        case TB_WIN:          score = TB_WIN_SCORE;    break;
        case TB_CURSED_WIN:   score = TB_CURSED_WIN;   break;
        case TB_DRAW:         score = 0;               break;
        case TB_BLESSED_LOSS: score = TB_BLESSED_LOSS; break;
        case TB_LOSS:         score = -TB_WIN_SCORE;   break;
        default: score = 0; break;
    }
    return true;
#else
    (void)board; (void)bestMove; (void)score;
    return false;
#endif
}

} // namespace Syzygy
