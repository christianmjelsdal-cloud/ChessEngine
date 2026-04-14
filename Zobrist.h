#pragma once
// ============================================================
//  Zobrist.h — Standalone Zobrist hashing tables
//
//  Separated from Engine so Board::applyMove can update
//  board.hash incrementally without a circular dependency.
//
//  Call Zobrist::init() once at startup (Engine::initZobrist
//  delegates here). All tables are global singletons.
// ============================================================
#include "Types.h"
#include <cstdint>
#include <random>

namespace Zobrist {

// Tables (defined in Engine.cpp via init())
extern uint64_t piece[2][7][64];  // [color][pieceType][square]
extern uint64_t castle[16];       // castling rights bitmask
extern uint64_t ep[8];            // en-passant file
extern uint64_t side;             // black to move
extern uint64_t duck[64];         // duck square
extern bool     ready;

inline void init() {
    if (ready) return;
    std::mt19937_64 rng(0xBEEF1234ULL);
    for (int c = 0; c < 2; c++)
        for (int p = 0; p < 7; p++)
            for (int s = 0; s < 64; s++)
                piece[c][p][s] = rng();
    for (int i = 0; i < 16; i++) castle[i] = rng();
    for (int i = 0; i < 8;  i++) ep[i]     = rng();
    side = rng();
    for (int i = 0; i < 64; i++) duck[i] = rng();
    ready = true;
}

} // namespace Zobrist
