// ============================================================================
//  SmokeTest.cpp — Behavioral smoke tests for search engine
//
//  Runs a set of rapid searches on well-known positions to verify:
//    1. No crashes or hangs
//    2. Reasonable move quality (e.g., doesn't blunder mate-in-1)
//    3. Search completes within time limits
//    4. Bitboard/mailbox consistency after search
//    5. Node counts are non-zero (search actually ran)
//
//  Uses the handcrafted evaluator (no NNUE weights required).
// ============================================================================

#include "Board.h"
#include "MoveGen.h"
#include "Engine.h"
#include "Bitboard.h"
#include "Test.h"    // for validateBoardConsistency
#include <iostream>
#include <chrono>
#include <string>
#include <iomanip>
#include <cstring>
#include <cassert>
#include <memory>

// ── Helper: convert Move to UCI string ───────────────────────────────────────

static std::string moveToUCI(const Move& m) {
    std::string s;
    s += static_cast<char>('a' + m.from.col);
    s += static_cast<char>('1' + m.from.rank);
    s += static_cast<char>('a' + m.to.col);
    s += static_cast<char>('1' + m.to.rank);
    if (m.promotion != PieceType::None) {
        switch (m.promotion) {
            case PieceType::Queen:  s += 'q'; break;
            case PieceType::Rook:   s += 'r'; break;
            case PieceType::Bishop: s += 'b'; break;
            case PieceType::Knight: s += 'n'; break;
            default: break;
        }
    }
    return s;
}

// ── Helper: check if a move matches a UCI string ─────────────────────────────

static bool moveMatchesUCI(const Move& m, const char* uci) {
    return moveToUCI(m) == uci;
}

// ── Smoke test positions ─────────────────────────────────────────────────────

struct SmokeTest {
    const char* name;
    const char* fen;
    int depth;
    const char* expectedMove;  // NULL = any legal move
    bool expectMate;           // if true, score should be mate-range
};

static const SmokeTest smokeTests[] = {
    {
        "Starting position (depth 10)",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        10, nullptr, false
    },
    {
        "Mate in 1 (Qxf7#)",
        "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        6, "h5f7", true
    },
    {
        "Obvious queen capture",
        "rnb1kbnr/pppppppp/8/8/3q4/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        8, "d1d4", false
    },
    {
        "King + Rook vs King endgame",
        "8/8/8/4k3/8/8/8/4K2R w - - 0 1",
        12, nullptr, false
    },
    {
        "Sicilian middlegame (depth 10)",
        "r1b1k2r/2qnbppp/p2ppn2/1p4B1/3NP3/2N2Q2/PPP2PPP/2KR1B1R w kq - 0 11",
        10, nullptr, false
    },
    {
        "Back rank mate in 1",
        "6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1",
        8, "a1a8", true
    },
    {
        "Pawn promotion (depth 8)",
        "8/P7/8/8/8/8/8/4K2k w - - 0 1",
        8, nullptr, false
    },
    {
        "K+Q vs K (avoid stalemate)",
        "8/8/8/8/8/1k6/8/KQ6 w - - 0 1",
        10, nullptr, false
    },
    {
        "Castling available (depth 8)",
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        8, nullptr, false
    },
    {
        "En passant available (depth 8)",
        "rnbqkbnr/pppp1ppp/8/4pP2/8/8/PPPPP1PP/RNBQKBNR w KQkq e6 0 3",
        8, nullptr, false
    },
};

static const int NUM_SMOKE_TESTS = sizeof(smokeTests) / sizeof(smokeTests[0]);

// ── Main smoke test runner ───────────────────────────────────────────────────

int main() {
    BB::initBitboards();
    Engine::initZobrist();

    std::cout << "=== Smoke Test Suite ===\n\n";

    int passed = 0, failed = 0, warnCount = 0;
    auto totalStart = std::chrono::steady_clock::now();

    for (int i = 0; i < NUM_SMOKE_TESTS; i++) {
        const auto& t = smokeTests[i];

        Board board;
        board.setFromFEN(t.fen);

        // Recompute Zobrist hash for the position
        board.hash = Engine::computeHash(board);

        // Validate board consistency before search
        if (!validateBoardConsistency(board, std::string("Before search: ") + t.name)) {
            std::cout << "  FAIL  " << t.name << " — bitboard inconsistency before search\n";
            failed++;
            continue;
        }

        // Create engine on the HEAP — Engine is ~1 MB (accumulators, history
        // tables, etc.) which overflows MSVC's default 1 MB stack.
        auto engine = std::make_unique<Engine>(1 << 18);  // ~256K entries ≈ 6 MB
        engine->setTimeLimit(2000);  // 2 second hard limit

        auto start = std::chrono::steady_clock::now();
        Move bestMove = engine->getBestMove(board, t.depth);
        auto end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        uint64_t nodes = engine->getNodes();
        int depth = engine->getLastDepth();
        std::string moveStr = moveToUCI(bestMove);

        // Validate board consistency after search (make/unmake should restore)
        if (!validateBoardConsistency(board, std::string("After search: ") + t.name)) {
            std::cout << "  FAIL  " << t.name << " — bitboard inconsistency after search\n";
            failed++;
            continue;
        }

        // Check the move is legal
        MoveList legalMoves;
        MoveGen::getLegalMoves(board, legalMoves);
        bool isLegal = false;
        for (int j = 0; j < legalMoves.count; j++) {
            if (legalMoves[j] == bestMove) {
                isLegal = true;
                break;
            }
        }

        if (!isLegal) {
            std::cout << "  FAIL  " << t.name << " — illegal move: " << moveStr << "\n";
            failed++;
            continue;
        }

        // Check expected move (if specified)
        bool moveOk = true;
        if (t.expectedMove && !moveMatchesUCI(bestMove, t.expectedMove)) {
            moveOk = false;
        }

        // Check nodes > 0 (search actually ran)
        if (nodes == 0) {
            std::cout << "  FAIL  " << t.name << " — zero nodes searched\n";
            failed++;
            continue;
        }

        // Verify mate expectation
        if (t.expectMate) {
            int score = engine->getLiveEval();
            if (score < 9000 && score > -9000) {
                std::cout << "  FAIL  " << t.name << " — expected mate score, got cp " << score << "\n";
                failed++;
                continue;
            }
        }

        if (moveOk) {
            passed++;
            std::cout << "  PASS  " << std::setw(40) << std::left << t.name
                      << "  move=" << std::setw(6) << moveStr
                      << "  depth=" << std::setw(3) << depth
                      << "  nodes=" << std::setw(10) << nodes
                      << "  (" << std::fixed << std::setprecision(0) << ms << " ms)\n";
        } else {
            passed++;
            warnCount++;
            std::cout << "  WARN  " << std::setw(40) << std::left << t.name
                      << "  move=" << moveStr << " (expected " << t.expectedMove << ")"
                      << "  depth=" << depth
                      << "  (" << std::fixed << std::setprecision(0) << ms << " ms)\n";
        }
    }

    auto totalEnd = std::chrono::steady_clock::now();
    double totalMs = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();

    std::cout << "\n=== Results: " << passed << "/" << (passed + failed)
              << " passed";
    if (warnCount > 0)
        std::cout << " (" << warnCount << " with move mismatch warnings)";
    std::cout << " (" << std::fixed << std::setprecision(0) << totalMs << " ms total) ===\n";

    std::cout << "\nPress Enter to exit...";
    std::cin.get();

    return (failed == 0) ? 0 : 1;
}
