#pragma once
#include "Board.h"
#include "MoveGen.h"
#include "NNUE.h"
#include <vector>
#include <cstdint>
#include <atomic>
#include <chrono>
#include <mutex>

class Engine {
public:
    Engine();

    /// Iterative-deepening search with time management.
    /// Searches until timeLimitMs expires or maxDepth is reached.
    Move getBestMove(Board& board, int maxDepth = 64);

    /// Signal the engine to abort the current search (call from another thread).
    void stop() { stop_.store(true, std::memory_order_relaxed); }

    /// Set time limit in milliseconds (default 3000 = 3 seconds)
    void setTimeLimit(int ms) { timeLimitMs_ = ms; }

    /// Get the principal variation from the last completed search.
    const std::vector<Move>& getPV() const { return lastPV_; }

    /// Get the depth reached in the last completed search.
    int getLastDepth() const { return lastDepth_; }

    /// Set position history for repetition detection (call before getBestMove).
    void setPositionHistory(const std::vector<uint64_t>& hashes) { gameHistory_ = hashes; }

    /// Get live PV (thread-safe — call from render thread during search).
    std::vector<Move> getLivePV();

    /// Get live search depth and eval (thread-safe — call from render thread).
    int getLiveDepth() const { return liveDepth_.load(std::memory_order_relaxed); }
    int getLiveEval()  const { return liveEval_.load(std::memory_order_relaxed); }

    /// Set/clear NNUE network for evaluation (nullptr = use handcrafted eval).
    void setNNUE(NNUE::Network* net) { nnue_ = net; }
    NNUE::Network* getNNUE() const { return nnue_; }

    /// Compute Zobrist hash for a position (public so VisualGame can track history).
    static uint64_t computeHash(const Board& board);

    /// Initialize Zobrist tables (called automatically by constructor).
    static void initZobrist();

private:
    /* ---------- constants ---------- */
    static constexpr int MATE_SCORE = 100000;
    static constexpr int INF        = MATE_SCORE + 1;
    static constexpr int MAX_PLY    = 64;
    static constexpr size_t TT_SIZE = 1 << 22;   // ~4M entries

    /* ---------- search ---------- */
    int search(Board& board, int depth, int alpha, int beta,
               int ply, bool doNull, bool isPV);
    int qsearch(Board& board, int alpha, int beta, int ply);

    /* ---------- evaluation ---------- */
    int evaluate(const Board& board);

    /* ---------- NNUE ---------- */
    NNUE::Network* nnue_ = nullptr;

    /* ---------- SEE (Static Exchange Evaluation) ---------- */
    static int see(const Board& board, const Move& m);
    static PieceType findLVA(const Board& board, Square target, Color side,
                             const bool removed[8][8], Square& outSq);

    /* ---------- move ordering ---------- */
    void orderMoves(std::vector<Move>& moves, const Board& board,
                    int ply, const Move& hashMove) const;
    int  scoreMove(const Move& m, const Board& board,
                   int ply, const Move& hashMove) const;
    static int mvvLva(const Board& board, const Move& m);
    static bool isCapture(const Board& board, const Move& m);

    /* ---------- Zobrist hashing ---------- */
    static bool     zobristReady_;
    static uint64_t zPiece_[2][7][64];
    static uint64_t zCastle_[16];
    static uint64_t zEP_[8];
    static uint64_t zSide_;

    // === Duck Chess Zobrist ===
    static uint64_t zDuck_[64];

    /* ---------- Duck Chess search ---------- */
    int searchDuck(Board& board, int depth, int alpha, int beta, int ply);
    int scoreDuckPlacement(const Board& board, Square duckSq, Color myColor) const;
    void orderDuckPlacements(std::vector<Square>& placements, const Board& board, Color myColor) const;

    /* ---------- transposition table ---------- */
    enum TTFlag : uint8_t { TT_EXACT, TT_LOWER, TT_UPPER };
    struct TTEntry {
        uint64_t key   = 0;
        int32_t  score = 0;
        int16_t  depth = -1;
        TTFlag   flag  = TT_UPPER;
        Move     best{};
        uint8_t  gen   = 0;   // generation for aging
    };
    std::vector<TTEntry> tt_;
    uint8_t ttGen_ = 0;       // incremented each getBestMove call

    /* ---------- killer moves (2 per ply) ---------- */
    Move killers_[MAX_PLY][2]{};

    /* ---------- history heuristic [color][fromSq][toSq] ---------- */
    int history_[2][64][64]{};

    /* ---------- countermove heuristic [color][fromSq][toSq] ---------- */
    Move countermoves_[2][64][64]{};
    Move previousMove_{};

    /* ---------- principal variation ---------- */
    Move pvTable_[MAX_PLY][MAX_PLY]{};
    int  pvLength_[MAX_PLY]{};
    std::vector<Move> lastPV_;   // PV from last completed iteration
    int lastDepth_ = 0;

    /* ---------- live PV for real-time arrow display ---------- */
    std::mutex livePVMutex_;
    std::vector<Move> livePV_;

    /* ---------- live depth & eval for UI display ---------- */
    std::atomic<int> liveDepth_{0};
    std::atomic<int> liveEval_{0};   // from White's perspective (centipawns)

    /* ---------- repetition detection ---------- */
    std::vector<uint64_t> gameHistory_;   // position hashes from the game
    uint64_t searchStack_[MAX_PLY]{};     // hashes along the search path

    /* ---------- dynamic draw scoring ---------- */
    // Instead of treating draws as 0, bias based on root evaluation:
    // - Winning side avoids draws (negative draw score)
    // - Losing side seeks draws (positive draw score)
    int rootEval_ = 0;   // evaluation at root, from side-to-move's perspective
    int drawScore() const;

    /* ---------- search control ---------- */
    std::atomic<bool> stop_{false};
    int nodes_ = 0;
    int timeLimitMs_ = 3000;
    std::chrono::steady_clock::time_point searchStart_;

    bool shouldStop();   // checks time + stop flag (called periodically)
};
