#pragma once
#include <cassert>

// TIER-2 FIX #10: Define USE_PEXT at compile time to use BMI2 PEXT-based sliding
// attack lookups instead of magic multiplication.  Faster on AMD Zen 3+ / Intel Haswell+.
// Build with: -DUSE_PEXT -mbmi2
#ifdef USE_PEXT
#include <immintrin.h>    // _pext_u64
#endif

#if defined(_MSC_VER)
#include <intrin.h>       // MSVC intrinsics: __popcnt64, _BitScanForward64, etc.
#endif
// ============================================================================
// Bitboard.h — Bitboard infrastructure for chess engine
//
// Provides:
//   - Bitboard type and constants (file/rank masks, etc.)
//   - Bit manipulation utilities (popcount, lsb, msb, etc.)
//   - Precomputed attack tables (knight, king, pawn)
//   - Between/Line/Ray bitboards for sliding piece logic
//   - Magic bitboard tables for bishop, rook, and queen attacks
//
// Square mapping convention (consistent with Types.h):
//   index = rank * 8 + col
//   rank 0 = white's back rank (rank 1), col 0 = a-file
//   So a1=0, b1=1, ... h1=7, a2=8, ... h8=63
// ============================================================================

#include <cstdint>
#include "Types.h"

using Bitboard = uint64_t;

namespace BB {

// ──────────────────────────────────────────────────────────────────────────
// Square ↔ index conversion helpers
// ──────────────────────────────────────────────────────────────────────────
inline int toSquareIndex(int rank, int col) { return rank * 8 + col; }
inline int toSquareIndex(const Square& s)   { return s.rank * 8 + s.col; }
inline int toRank(int sq) { return sq / 8; }  // a.k.a. row
inline int toCol(int sq)  { return sq % 8; }
inline Square toSquare(int sq) { return { sq / 8, sq % 8 }; }

// ──────────────────────────────────────────────────────────────────────────
// Constants — file masks
// ──────────────────────────────────────────────────────────────────────────
constexpr Bitboard FileA = 0x0101010101010101ULL;
constexpr Bitboard FileB = FileA << 1;
constexpr Bitboard FileC = FileA << 2;
constexpr Bitboard FileD = FileA << 3;
constexpr Bitboard FileE = FileA << 4;
constexpr Bitboard FileF = FileA << 5;
constexpr Bitboard FileG = FileA << 6;
constexpr Bitboard FileH = FileA << 7;

// ──────────────────────────────────────────────────────────────────────────
// Constants — rank masks
// ──────────────────────────────────────────────────────────────────────────
constexpr Bitboard Rank1 = 0xFFULL;
constexpr Bitboard Rank2 = Rank1 << 8;
constexpr Bitboard Rank3 = Rank1 << 16;
constexpr Bitboard Rank4 = Rank1 << 24;
constexpr Bitboard Rank5 = Rank1 << 32;
constexpr Bitboard Rank6 = Rank1 << 40;
constexpr Bitboard Rank7 = Rank1 << 48;
constexpr Bitboard Rank8 = Rank1 << 56;

// ──────────────────────────────────────────────────────────────────────────
// Constants — misc
// ──────────────────────────────────────────────────────────────────────────
constexpr Bitboard Empty = 0ULL;
constexpr Bitboard Full  = ~0ULL;
constexpr Bitboard DarkSquares  = 0xAA55AA55AA55AA55ULL;
constexpr Bitboard LightSquares = 0x55AA55AA55AA55AAULL;

// ──────────────────────────────────────────────────────────────────────────
// Bit manipulation
// ──────────────────────────────────────────────────────────────────────────

/// Population count — number of set bits.
inline int popcount(Bitboard b) {
#if defined(_MSC_VER)
    return static_cast<int>(__popcnt64(b));
#else
    return __builtin_popcountll(b);
#endif
}

/// Index of least significant set bit.  Undefined if b == 0.
inline int lsb(Bitboard b) {
    assert(b != 0 && "lsb() called with b == 0 (undefined behavior)");  // 5.17
#if defined(_MSC_VER)
    unsigned long idx;
    _BitScanForward64(&idx, b);
    return static_cast<int>(idx);
#else
    return __builtin_ctzll(b);
#endif
}

/// Index of most significant set bit.  Undefined if b == 0.
inline int msb(Bitboard b) {
    assert(b != 0 && "msb() called with b == 0 (undefined behavior)");  // 5.17
#if defined(_MSC_VER)
    unsigned long idx;
    _BitScanReverse64(&idx, b);
    return static_cast<int>(idx);
#else
    return 63 - __builtin_clzll(b);
#endif
}

/// Pop the least significant bit and return its index.
inline int popLsb(Bitboard& b) {
    int s = lsb(b);
    b &= b - 1;          // clear the lsb
    return s;
}

/// True if more than one bit is set.
inline bool moreThanOne(Bitboard b) {
    return b & (b - 1);
}

/// Single-bit bitboard for a given square index (rank*8 + col).
inline Bitboard squareBB(int sq) {
    return 1ULL << sq;
}

/// Single-bit bitboard for a given (rank, col) pair.
inline Bitboard squareBB(int rank, int col) {
    return 1ULL << (rank * 8 + col);
}

/// Single-bit bitboard from a Square struct.
inline Bitboard squareBB(const Square& s) {
    return 1ULL << (s.rank * 8 + s.col);
}

// ──────────────────────────────────────────────────────────────────────────
// Precomputed attack tables (defined in Bitboard.cpp)
// ──────────────────────────────────────────────────────────────────────────
extern Bitboard KnightAttacks[64];
extern Bitboard KingAttacks[64];
extern Bitboard PawnAttacks[2][64];   // [Color::White=0 / Color::Black=1][sq]
extern Bitboard BetweenBB[64][64];    // squares strictly between two squares on a line
extern Bitboard LineBB[64][64];       // full line through two aligned squares
extern Bitboard RayBB[64][8];         // rays in 8 directions from sq

// Direction indices for RayBB
enum Direction : int {
    North = 0, NorthEast = 1, East = 2, SouthEast = 3,
    South = 4, SouthWest = 5, West = 6, NorthWest = 7
};

// ──────────────────────────────────────────────────────────────────────────
// Magic Bitboard infrastructure for sliding-piece attacks
//
// "Plain" magic bitboards (no PEXT/BMI2 for portability).
//
// For each square, the relevant occupancy bits (inner squares of the ray,
// excluding edges) are extracted with a mask, multiplied by a "magic" number,
// and shifted right to produce an index into a precomputed attack table.
//
//   index = ((occ & mask) * magic) >> shift
//   attacks = table[index]
//
// The magic numbers are chosen so that all relevant occupancy patterns for
// a given square hash to distinct indices (a perfect or semi-perfect hash).
// ──────────────────────────────────────────────────────────────────────────

struct MagicEntry {
    Bitboard  mask;     // relevant occupancy mask (excludes board edges)
    Bitboard  magic;    // magic multiplier
    Bitboard* attacks;  // pointer into a large precomputed attack table
    int       shift;    // 64 - popcount(mask) = right-shift amount
};

extern MagicEntry BishopMagics[64];
extern MagicEntry RookMagics[64];

/// Bishop attack set for a given square and board occupancy.
inline Bitboard bishopAttacks(int sq, Bitboard occ) {
    const auto& m = BishopMagics[sq];
#ifdef USE_PEXT
    // TIER-2 FIX #10: PEXT gives a dense perfect index — no magic multiply needed
    return m.attacks[_pext_u64(occ, m.mask)];
#else
    return m.attacks[((occ & m.mask) * m.magic) >> m.shift];
#endif
}

/// Rook attack set for a given square and board occupancy.
inline Bitboard rookAttacks(int sq, Bitboard occ) {
    const auto& m = RookMagics[sq];
#ifdef USE_PEXT
    return m.attacks[_pext_u64(occ, m.mask)];
#else
    return m.attacks[((occ & m.mask) * m.magic) >> m.shift];
#endif
}

/// Queen attack set = union of bishop and rook attacks.
inline Bitboard queenAttacks(int sq, Bitboard occ) {
    return bishopAttacks(sq, occ) | rookAttacks(sq, occ);
}

// ──────────────────────────────────────────────────────────────────────────
// Convenience: attack lookups from Square struct
// ──────────────────────────────────────────────────────────────────────────
inline Bitboard knightAttacks(int sq)                  { return KnightAttacks[sq]; }
inline Bitboard kingAttacks(int sq)                    { return KingAttacks[sq]; }
inline Bitboard pawnAttacks(Color c, int sq)           { return PawnAttacks[static_cast<int>(c)][sq]; }
inline Bitboard knightAttacks(const Square& s)         { return KnightAttacks[toSquareIndex(s)]; }
inline Bitboard kingAttacks(const Square& s)           { return KingAttacks[toSquareIndex(s)]; }
inline Bitboard pawnAttacks(Color c, const Square& s)  { return PawnAttacks[static_cast<int>(c)][toSquareIndex(s)]; }
inline Bitboard bishopAttacks(const Square& s, Bitboard occ) { return bishopAttacks(toSquareIndex(s), occ); }
inline Bitboard rookAttacks(const Square& s, Bitboard occ)   { return rookAttacks(toSquareIndex(s), occ); }
inline Bitboard queenAttacks(const Square& s, Bitboard occ)  { return queenAttacks(toSquareIndex(s), occ); }

// ──────────────────────────────────────────────────────────────────────────
// Initialization — must be called once at program startup before any
// bitboard lookup is used.
// ──────────────────────────────────────────────────────────────────────────
void initBitboards();

} // namespace BB
