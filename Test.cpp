// ============================================================================
//  Test.cpp — Perft test suite for move generation validation
//
//  Validates Board + MoveGen correctness using standard perft positions with
//  externally verified reference node counts. Also checks bitboard/mailbox
//  consistency at every node to catch representation bugs.
//
//  Reference values from: https://www.chessprogramming.org/Perft_Results
// ============================================================================

#include "Test.h"
#include "Bitboard.h"
#include "Engine.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <cassert>

// ── Perft implementation ─────────────────────────────────────────────────────

uint64_t perft(Board& board, int depth) {
    if (depth == 0) return 1;

    MoveList moves;
    MoveGen::getLegalMoves(board, moves);

    if (depth == 1) return static_cast<uint64_t>(moves.count);

    uint64_t nodes = 0;
    for (int i = 0; i < moves.count; i++) {
#ifndef NDEBUG
        // AUDIT 3.21: Verify applyMove produces same state as makeMove
        assert(verifyApplyMakeEquivalence(board, moves[i])
            && "applyMove and makeMove produced different board states");
#endif
        Board::UndoInfo undo;
        board.makeMove(moves[i], undo);
        nodes += perft(board, depth - 1);
        board.unmakeMove(moves[i], undo);
    }
    return nodes;
}

// ── Board consistency check ──────────────────────────────────────────────────

bool validateBoardConsistency(const Board& board, const std::string& context) {
    // FIX 5.1: Rebuild bitboards from mailbox and compare (including Duck slot)
    const int BB_COUNT = (int)(sizeof(board.pieceBBs) / sizeof(board.pieceBBs[0]));
    uint64_t pieceBB[BB_COUNT] = {};
    uint64_t colorBB[2] = {};
    uint64_t occupiedBB = 0;

    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            Piece p = board.squares[r][c];
            if (p.isNone()) continue;
            int sq = r * 8 + c;
            uint64_t mask = 1ULL << sq;
            occupiedBB |= mask;
#ifdef DUCK_CHESS
            if (p.isDuck()) {
                pieceBB[static_cast<int>(PieceType::Duck)] |= mask;
                continue;
            }
#endif
            pieceBB[static_cast<int>(p.type)] |= mask;
            colorBB[static_cast<int>(p.color)] |= mask;
        }
    }

    bool ok = true;
    for (int i = 0; i < BB_COUNT; i++) {
        if (board.pieceBBs[i] != pieceBB[i]) {
            std::cerr << "  MISMATCH pieceBB[" << i << "]: board=0x"
                      << std::hex << board.pieceBBs[i] << " expected=0x"
                      << pieceBB[i] << std::dec << "\n";
            ok = false;
        }
    }
    for (int i = 0; i < 2; i++) {
        if (board.colorBB[i] != colorBB[i]) {
            std::cerr << "  MISMATCH colorBB[" << i << "]: board=0x"
                      << std::hex << board.colorBB[i] << " expected=0x"
                      << colorBB[i] << std::dec << "\n";
            ok = false;
        }
    }
    if (board.occupiedBB != occupiedBB) {
        std::cerr << "  MISMATCH occupiedBB: board=0x"
                  << std::hex << board.occupiedBB << " expected=0x"
                  << occupiedBB << std::dec << "\n";
        ok = false;
    }

    if (!ok && !context.empty()) {
        std::cerr << "  Context: " << context << "\n";
    }
    return ok;
}

// ── applyMove vs makeMove verification (debug builds) ───────────────────────
// AUDIT 3.21: Verify that applyMove and makeMove produce identical board state.
// This catches divergence between the GUI path (applyMove) and search path
// (makeMove) that perft alone would not detect.

#ifndef NDEBUG
bool verifyApplyMakeEquivalence(const Board& original, const Move& move) {
    // Path 1: applyMove on a copy
    Board applied = original;
    applied.applyMove(move);

    // Path 2: makeMove + unmakeMove round-trip, then makeMove again
    Board maked = original;
    Board::UndoInfo undo;
    maked.makeMove(move, undo);

    // Compare the resulting board states
    if (!(applied == maked)) {
        std::cerr << "applyMove/makeMove MISMATCH for move ("
                  << move.from.rank << "," << move.from.col << ")->(" 
                  << move.to.rank << "," << move.to.col << ")\n";
        return false;
    }

    // Also verify round-trip: unmakeMove should restore the original
    maked.unmakeMove(move, undo);
    if (!(maked == original)) {
        std::cerr << "makeMove/unmakeMove round-trip MISMATCH for move ("
                  << move.from.rank << "," << move.from.col << ")->(" 
                  << move.to.rank << "," << move.to.col << ")\n";
        return false;
    }
    return true;
}
#endif

// ── Perft test positions ─────────────────────────────────────────────────────

struct PerftTest {
    const char* name;
    const char* fen;
    int depth;
    uint64_t expected;
};

static const PerftTest perftTests[] = {
    // Position 1: Starting position
    { "Startpos depth 1", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 1, 20 },
    { "Startpos depth 2", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 2, 400 },
    { "Startpos depth 3", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 3, 8902 },
    { "Startpos depth 4", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 4, 197281 },
    { "Startpos depth 5", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 5, 4865609 },

    // Position 2: "Kiwipete" — complex position with many special moves
    { "Kiwipete depth 1", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 1, 48 },
    { "Kiwipete depth 2", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 2, 2039 },
    { "Kiwipete depth 3", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3, 97862 },
    { "Kiwipete depth 4", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4085603 },

    // Position 3: En passant + pawn edge cases
    { "Position 3 depth 1", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 1, 14 },
    { "Position 3 depth 2", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 2, 191 },
    { "Position 3 depth 3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 3, 2812 },
    { "Position 3 depth 4", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 4, 43238 },
    { "Position 3 depth 5", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5, 674624 },

    // Position 4: Promotions + castling rights stress
    { "Position 4 depth 1", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 1, 6 },
    { "Position 4 depth 2", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 2, 264 },
    { "Position 4 depth 3", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 3, 9467 },
    { "Position 4 depth 4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 4, 422333 },

    // Position 5: Promotion on first rank capture
    { "Position 5 depth 1", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 1, 44 },
    { "Position 5 depth 2", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 2, 1486 },
    { "Position 5 depth 3", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 3, 62379 },
    { "Position 5 depth 4", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 4, 2103487 },

    // Position 6: Alternative promotion test (CPW)
    { "Position 6 depth 1", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 1, 46 },
    { "Position 6 depth 2", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 2, 2079 },
    { "Position 6 depth 3", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 3, 89890 },
    { "Position 6 depth 4", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 4, 3894594 },
};

static const int NUM_PERFT_TESTS = sizeof(perftTests) / sizeof(perftTests[0]);

// ── Main test runner ─────────────────────────────────────────────────────────

// INFO [11.19]: No edge-case tests (double-check, EP pin, castling blocked, 50-move, stalemate promo).
int runPerftSuite() {
    int passed = 0, failed = 0;
    auto totalStart = std::chrono::steady_clock::now();

    std::cout << "=== Perft Test Suite ===\n\n";

    for (int i = 0; i < NUM_PERFT_TESTS; i++) {
        const auto& t = perftTests[i];
        Board board;
        board.setFromFEN(t.fen);

        // Validate bitboard consistency after FEN parse
        if (!validateBoardConsistency(board, std::string("After FEN parse: ") + t.name)) {
            std::cout << "  FAIL (bitboard inconsistency after FEN parse)\n";
            failed++;
            continue;
        }

        auto start = std::chrono::steady_clock::now();
        uint64_t result = perft(board, t.depth);
        auto end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        // Validate bitboard consistency after perft (make/unmake should restore perfectly)
        if (!validateBoardConsistency(board, std::string("After perft: ") + t.name)) {
            std::cout << "  FAIL (bitboard inconsistency after make/unmake cycle)\n";
            failed++;
            continue;
        }

        bool ok = (result == t.expected);
        if (ok) {
            passed++;
            std::cout << "  PASS  " << std::setw(25) << std::left << t.name
                      << "  nodes=" << std::setw(10) << result
                      << "  (" << std::fixed << std::setprecision(1) << ms << " ms)\n";
        } else {
            failed++;
            std::cout << "  FAIL  " << std::setw(25) << std::left << t.name
                      << "  got=" << result << " expected=" << t.expected
                      << "  (" << std::fixed << std::setprecision(1) << ms << " ms)\n";
        }
    }

    auto totalEnd = std::chrono::steady_clock::now();
    double totalMs = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();

    std::cout << "\n=== Results: " << passed << "/" << (passed + failed)
              << " passed (" << std::fixed << std::setprecision(1) << totalMs << " ms total) ===\n";

    // INFO [3.24]: Interactive wait blocks CI/CD. Define CI_BUILD to skip.
#ifndef CI_BUILD
    std::cout << "\nPress Enter to exit...";
    std::cin.get();
#endif

    return (failed == 0) ? 0 : 1;
}

// ── Entry point ──────────────────────────────────────────────────────────────
// Guarded so Test.cpp can also be compiled as a library (TEST_NO_MAIN)
// for linking into SmokeTest.cpp without a duplicate main().

#ifndef TEST_NO_MAIN
int main() {
    // Initialize subsystems
    BB::initBitboards();
    Engine::initZobrist();

    return runPerftSuite();
}
#endif
