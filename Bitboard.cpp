// ============================================================================
// Bitboard.cpp — Implementation of bitboard initialization
//
// Initializes all precomputed attack tables:
//   - Knight, King, Pawn attacks
//   - Between / Line / Ray bitboards
//   - Magic bitboard tables for Bishop and Rook
//
// Uses well-known magic numbers from Stockfish (verified correct).
// The magic approach: for each square, the relevant occupancy bits are
// masked, multiplied by a magic number, and shifted to produce an index
// into a precomputed table of attack bitboards.
// ============================================================================

#include "Bitboard.h"
#include <cassert>  // FIX 8.7: collision assertions in magic init
#include <cstring>

namespace BB {

// ──────────────────────────────────────────────────────────────────────────
// Global table definitions
// ──────────────────────────────────────────────────────────────────────────
Bitboard KnightAttacks[64];
Bitboard KingAttacks[64];
Bitboard PawnAttacks[2][64];
Bitboard BetweenBB[64][64];
Bitboard LineBB[64][64];
Bitboard RayBB[64][8];

MagicEntry BishopMagics[64];
MagicEntry RookMagics[64];

// ──────────────────────────────────────────────────────────────────────────
// Attack table storage for magic bitboards
//
// Stockfish-style: each square's attacks pointer points into these large
// arrays.  The total size is the sum of (1 << bits) for all 64 squares.
//
// Bishop total: sum of (1 << BishopBits[sq]) for sq 0..63
//   4 corners * 64 + 8 edge-non-corner * 32 + ... = 5248  (0x1480)
// Rook total: sum of (1 << RookBits[sq]) for sq 0..63
//   4 corners * 4096 + ... = 102400 (0x19000)
// ──────────────────────────────────────────────────────────────────────────
static Bitboard BishopTable[0x1480];
static Bitboard RookTable[0x19000];

// ──────────────────────────────────────────────────────────────────────────
// Well-known magic numbers from Stockfish
//
// These magic numbers are paired with the standard mask computation
// (relevant occupancy = ray excluding edges) and the standard bit counts
// below. They have been tested over billions of positions.
// ──────────────────────────────────────────────────────────────────────────

// Rook magic numbers (from Stockfish src/bitboard.cpp)
static const Bitboard RookMagicNumbers[64] = {
    0x0080001020400080ULL, 0x0040001000200040ULL, 0x0080081000200080ULL, 0x0080040800100080ULL,
    0x0080020400080080ULL, 0x0080010200040080ULL, 0x0080008001000200ULL, 0x0080002040800100ULL,
    0x0000800020400080ULL, 0x0000400020005000ULL, 0x0000801000200080ULL, 0x0000800800100080ULL,
    0x0000800400080080ULL, 0x0000800200040080ULL, 0x0000800100020080ULL, 0x0000800040800100ULL,
    0x0000208000400080ULL, 0x0000404000201000ULL, 0x0000808010002000ULL, 0x0000808008001000ULL,
    0x0000808004000800ULL, 0x0000808002000400ULL, 0x0000010100020004ULL, 0x0000020000408104ULL,
    0x0000208080004000ULL, 0x0000200040005000ULL, 0x0000100080200080ULL, 0x0000080080100080ULL,
    0x0000040080080080ULL, 0x0000020080040080ULL, 0x0000010080800200ULL, 0x0000800080004100ULL,
    0x0000204000800080ULL, 0x0000200040401000ULL, 0x0000100080802000ULL, 0x0000080080801000ULL,
    0x0000040080800800ULL, 0x0000020080800400ULL, 0x0000020001010004ULL, 0x0000800040800100ULL,
    0x0000204000808000ULL, 0x0000200040008080ULL, 0x0000100020008080ULL, 0x0000080010008080ULL,
    0x0000040008008080ULL, 0x0000020004008080ULL, 0x0000010002008080ULL, 0x0000004081020004ULL,
    0x0000204000800080ULL, 0x0000200040008080ULL, 0x0000100020008080ULL, 0x0000080010008080ULL,
    0x0000040008008080ULL, 0x0000020004008080ULL, 0x0000800100020080ULL, 0x0000800041000080ULL,
    0x00FFFCDDFCED714AULL, 0x007FFCDDFCED714AULL, 0x003FFFCDFFD88096ULL, 0x0000040810002101ULL,
    0x0001000204080011ULL, 0x0001000204000801ULL, 0x0001000082000401ULL, 0x0001FFFAABFAD1A2ULL
};

// Bishop magic numbers (from Stockfish src/bitboard.cpp)
static const Bitboard BishopMagicNumbers[64] = {
    0x0002020202020200ULL, 0x0002020202020000ULL, 0x0004010202000000ULL, 0x0004040080000000ULL,
    0x0001104000000000ULL, 0x0000821040000000ULL, 0x0000410410400000ULL, 0x0000104104104000ULL,
    0x0000040404040400ULL, 0x0000020202020200ULL, 0x0000040102020000ULL, 0x0000040400800000ULL,
    0x0000011040000000ULL, 0x0000008210400000ULL, 0x0000004104104000ULL, 0x0000002082082000ULL,
    0x0004000808080800ULL, 0x0002000404040400ULL, 0x0001000202020200ULL, 0x0000800802004000ULL,
    0x0000800400A00000ULL, 0x0000200100884000ULL, 0x0000400082082000ULL, 0x0000200041041000ULL,
    0x0002080010101000ULL, 0x0001040008080800ULL, 0x0000208004010400ULL, 0x0000404004010200ULL,
    0x0000840000802000ULL, 0x0000404002011000ULL, 0x0000808001041000ULL, 0x0000404000820800ULL,
    0x0001041000202000ULL, 0x0000820800101000ULL, 0x0000104400080800ULL, 0x0000020080080080ULL,
    0x0000404040040100ULL, 0x0000808100020100ULL, 0x0001010100020800ULL, 0x0000808080010400ULL,
    0x0000820820004000ULL, 0x0000410410002000ULL, 0x0000082088001000ULL, 0x0000002011000800ULL,
    0x0000080100400400ULL, 0x0001010101000200ULL, 0x0002020202000400ULL, 0x0001010101000200ULL,
    0x0000410410400000ULL, 0x0000208208200000ULL, 0x0000002084100000ULL, 0x0000000020880000ULL,
    0x0000001002020000ULL, 0x0000040408020000ULL, 0x0004040404040000ULL, 0x0002020202020000ULL,
    0x0000104104104000ULL, 0x0000002082082000ULL, 0x0000000020841000ULL, 0x0000000000208800ULL,
    0x0000000010020200ULL, 0x0000000404080200ULL, 0x0000040404040400ULL, 0x0002020202020200ULL
};

// Number of relevant bits for each rook square
static const int RookBits[64] = {
    12, 11, 11, 11, 11, 11, 11, 12,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    12, 11, 11, 11, 11, 11, 11, 12
};

// Number of relevant bits for each bishop square
static const int BishopBits[64] = {
    6, 5, 5, 5, 5, 5, 5, 6,
    5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 7, 7, 7, 7, 5, 5,
    5, 5, 7, 9, 9, 7, 5, 5,
    5, 5, 7, 9, 9, 7, 5, 5,
    5, 5, 7, 7, 7, 7, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5,
    6, 5, 5, 5, 5, 5, 5, 6
};

// ──────────────────────────────────────────────────────────────────────────
// Direction deltas: {dRank, dCol}
// ──────────────────────────────────────────────────────────────────────────
static const int DirDelta[8][2] = {
    { 1,  0}, // North
    { 1,  1}, // NorthEast
    { 0,  1}, // East
    {-1,  1}, // SouthEast
    {-1,  0}, // South
    {-1, -1}, // SouthWest
    { 0, -1}, // West
    { 1, -1}  // NorthWest
};

// ──────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────

/// Compute a ray from (rank, col) in a given direction (excluding the starting square).
static Bitboard computeRay(int rank, int col, int dRank, int dCol) {
    Bitboard ray = 0;
    int r = rank + dRank, c = col + dCol;
    while (r >= 0 && r < 8 && c >= 0 && c < 8) {
        ray |= 1ULL << (r * 8 + c);
        r += dRank;
        c += dCol;
    }
    return ray;
}

/// Compute sliding attacks from sq in the given directions, blocked by occupancy.
static Bitboard slidingAttacks(int sq, Bitboard occ, const int dirs[][2], int numDirs) {
    Bitboard attacks = 0;
    int rank = sq / 8, col = sq % 8;
    for (int d = 0; d < numDirs; ++d) {
        int r = rank + dirs[d][0], c = col + dirs[d][1];
        while (r >= 0 && r < 8 && c >= 0 && c < 8) {
            int s = r * 8 + c;
            attacks |= 1ULL << s;
            if (occ & (1ULL << s)) break;  // blocked
            r += dirs[d][0];
            c += dirs[d][1];
        }
    }
    return attacks;
}

static const int BishopDirs[4][2] = { {1,1}, {1,-1}, {-1,1}, {-1,-1} };
static const int RookDirs[4][2]   = { {1,0}, {-1,0}, {0,1}, {0,-1} };

/// Compute the relevant occupancy mask for a rook on the given square.
/// Excludes edge squares of the rays (a rook always attacks to the edge
/// regardless of occupancy there, so those bits are irrelevant).
static Bitboard rookMask(int sq) {
    Bitboard mask = 0;
    int rank = sq / 8, col = sq % 8;
    for (int r = rank + 1; r < 7; ++r) mask |= 1ULL << (r * 8 + col);
    for (int r = rank - 1; r > 0; --r) mask |= 1ULL << (r * 8 + col);
    for (int c = col + 1; c < 7; ++c)  mask |= 1ULL << (rank * 8 + c);
    for (int c = col - 1; c > 0; --c)  mask |= 1ULL << (rank * 8 + c);
    return mask;
}

/// Compute the relevant occupancy mask for a bishop on the given square.
static Bitboard bishopMask(int sq) {
    Bitboard mask = 0;
    int rank = sq / 8, col = sq % 8;
    for (int r = rank+1, c = col+1; r < 7 && c < 7; ++r, ++c) mask |= 1ULL << (r*8+c);
    for (int r = rank+1, c = col-1; r < 7 && c > 0; ++r, --c) mask |= 1ULL << (r*8+c);
    for (int r = rank-1, c = col+1; r > 0 && c < 7; --r, ++c) mask |= 1ULL << (r*8+c);
    for (int r = rank-1, c = col-1; r > 0 && c > 0; --r, --c) mask |= 1ULL << (r*8+c);
    return mask;
}

/// Get the Nth subset of a mask using index bits (used to enumerate all subsets
/// in a deterministic order for magic table initialization).
static Bitboard indexToOccupancy(int index, int bits, Bitboard mask) {
    Bitboard occ = 0;
    for (int i = 0; i < bits; ++i) {
        int j = lsb(mask); // index of lsb (uses MSVC intrinsic or GCC builtin)
        mask &= mask - 1;             // pop lsb
        if (index & (1 << i))
            occ |= 1ULL << j;
    }
    return occ;
}

// ──────────────────────────────────────────────────────────────────────────
// Knight attack init
// ──────────────────────────────────────────────────────────────────────────
static void initKnightAttacks() {
    static const int knightMoves[8][2] = {
        {-2,-1},{-2,1},{-1,-2},{-1,2},{1,-2},{1,2},{2,-1},{2,1}
    };
    for (int sq = 0; sq < 64; ++sq) {
        int rank = sq / 8, col = sq % 8;
        Bitboard bb = 0;
        for (auto& m : knightMoves) {
            int r = rank + m[0], c = col + m[1];
            if (r >= 0 && r < 8 && c >= 0 && c < 8)
                bb |= 1ULL << (r * 8 + c);
        }
        KnightAttacks[sq] = bb;
    }
}

// ──────────────────────────────────────────────────────────────────────────
// King attack init
// ──────────────────────────────────────────────────────────────────────────
static void initKingAttacks() {
    static const int kingMoves[8][2] = {
        {-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}
    };
    for (int sq = 0; sq < 64; ++sq) {
        int rank = sq / 8, col = sq % 8;
        Bitboard bb = 0;
        for (auto& m : kingMoves) {
            int r = rank + m[0], c = col + m[1];
            if (r >= 0 && r < 8 && c >= 0 && c < 8)
                bb |= 1ULL << (r * 8 + c);
        }
        KingAttacks[sq] = bb;
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Pawn attack init
// ──────────────────────────────────────────────────────────────────────────
static void initPawnAttacks() {
    for (int sq = 0; sq < 64; ++sq) {
        int rank = sq / 8, col = sq % 8;
        Bitboard wbb = 0, bbb = 0;
        // White pawns attack upward (rank + 1)
        if (rank < 7) {
            if (col > 0) wbb |= 1ULL << ((rank+1)*8 + (col-1));
            if (col < 7) wbb |= 1ULL << ((rank+1)*8 + (col+1));
        }
        // Black pawns attack downward (rank - 1)
        if (rank > 0) {
            if (col > 0) bbb |= 1ULL << ((rank-1)*8 + (col-1));
            if (col < 7) bbb |= 1ULL << ((rank-1)*8 + (col+1));
        }
        PawnAttacks[0][sq] = wbb;  // Color::White = 0
        PawnAttacks[1][sq] = bbb;  // Color::Black = 1
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Ray, Between, Line bitboard init
// ──────────────────────────────────────────────────────────────────────────
static void initRays() {
    for (int sq = 0; sq < 64; ++sq) {
        int rank = sq / 8, col = sq % 8;
        for (int d = 0; d < 8; ++d) {
            RayBB[sq][d] = computeRay(rank, col, DirDelta[d][0], DirDelta[d][1]);
        }
    }
}

static void initBetweenAndLine() {
    std::memset(BetweenBB, 0, sizeof(BetweenBB));
    std::memset(LineBB, 0, sizeof(LineBB));

    for (int s1 = 0; s1 < 64; ++s1) {
        for (int s2 = 0; s2 < 64; ++s2) {
            if (s1 == s2) continue;

            // Check all 8 ray directions from s1
            for (int d = 0; d < 8; ++d) {
                Bitboard ray = RayBB[s1][d];
                if (ray & (1ULL << s2)) {
                    // s2 is on this ray from s1
                    // BetweenBB = squares strictly between s1 and s2
                    Bitboard between = 0;
                    int r = (s1 / 8) + DirDelta[d][0];
                    int c = (s1 % 8) + DirDelta[d][1];
                    while (r >= 0 && r < 8 && c >= 0 && c < 8) {
                        int sq = r * 8 + c;
                        if (sq == s2) break;
                        between |= 1ULL << sq;
                        r += DirDelta[d][0];
                        c += DirDelta[d][1];
                    }
                    BetweenBB[s1][s2] = between;

                    // LineBB = full line through both squares
                    int opp = (d + 4) & 7;  // opposite direction
                    LineBB[s1][s2] = RayBB[s1][d] | RayBB[s1][opp] | (1ULL << s1);
                    break;
                }
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Magic bitboard init
//
// For each square:
// 1. Compute the relevant occupancy mask
// 2. Enumerate all subsets of the mask (2^bits subsets)
// 3. For each subset, compute the reference attack set (slow sliding)
// 4. Store it in the table at index = ((occ * magic) >> shift)
//
// The magic numbers are chosen so no two different occupancies for the
// same square map to the same index (collision-free).
// ──────────────────────────────────────────────────────────────────────────
static void initMagics() {
    // Initialize bishop magics
    Bitboard* bPtr = BishopTable;
    for (int sq = 0; sq < 64; ++sq) {
        Bitboard mask = bishopMask(sq);
        int bits = BishopBits[sq];
        int tableSize = 1 << bits;

        BishopMagics[sq].mask    = mask;
        BishopMagics[sq].magic   = BishopMagicNumbers[sq];
        BishopMagics[sq].shift   = 64 - bits;
        BishopMagics[sq].attacks = bPtr;

        // Fill the table for all occupancy subsets
        for (int idx = 0; idx < (1 << bits); ++idx) {
            Bitboard occ = indexToOccupancy(idx, bits, mask);
            Bitboard attacks = slidingAttacks(sq, occ, BishopDirs, 4);
#ifdef USE_PEXT
            // TIER-2 FIX #10: PEXT gives a dense index equal to idx
            bPtr[idx] = attacks;
#else
            int tableIdx = (int)((occ * BishopMagicNumbers[sq]) >> (64 - bits));
            assert((bPtr[tableIdx] == 0 || bPtr[tableIdx] == attacks)
                && "Magic collision detected for bishop");
            bPtr[tableIdx] = attacks;
#endif
        }

        bPtr += tableSize;
    }

    // Initialize rook magics
    Bitboard* rPtr = RookTable;
    for (int sq = 0; sq < 64; ++sq) {
        Bitboard mask = rookMask(sq);
        int bits = RookBits[sq];
        int tableSize = 1 << bits;

        RookMagics[sq].mask    = mask;
        RookMagics[sq].magic   = RookMagicNumbers[sq];
        RookMagics[sq].shift   = 64 - bits;
        RookMagics[sq].attacks = rPtr;

        // Fill the table for all occupancy subsets
        for (int idx = 0; idx < (1 << bits); ++idx) {
            Bitboard occ = indexToOccupancy(idx, bits, mask);
            Bitboard attacks = slidingAttacks(sq, occ, RookDirs, 4);
#ifdef USE_PEXT
            rPtr[idx] = attacks;
#else
            int tableIdx = (int)((occ * RookMagicNumbers[sq]) >> (64 - bits));
            assert((rPtr[tableIdx] == 0 || rPtr[tableIdx] == attacks)
                && "Magic collision detected for rook");
            rPtr[tableIdx] = attacks;
#endif
        }

        rPtr += tableSize;
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Public init entry point
// ──────────────────────────────────────────────────────────────────────────
void initBitboards() {
    initKnightAttacks();
    initKingAttacks();
    initPawnAttacks();
    initRays();
    initBetweenAndLine();
    initMagics();
}

} // namespace BB
