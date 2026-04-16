#pragma once
#include "Board.h"
#include "MoveGen.h"
#include "NNUE.h"
#include "DuckNNUE.h"
#include <vector>
#include <cstdint>
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <functional>

class Engine {
public:
    Engine();
    explicit Engine(size_t ttSize);

    /// Iterative-deepening search with time management.
    Move getBestMove(Board& board, int maxDepth = 64);

    void stop() { stop_.store(true, std::memory_order_relaxed); }

    void setTimeLimit(int ms) { softLimitMs_ = ms; hardLimitMs_ = ms; }
    void setTimeLimits(int softMs, int hardMs) { softLimitMs_ = softMs; hardLimitMs_ = hardMs; }

    const std::vector<Move>& getPV() const { return lastPV_; }
    int      getLastDepth() const { return lastDepth_; }
    uint64_t getNodes()     const { return nodes_; }

    void setPositionHistory(const std::vector<uint64_t>& hashes) { gameHistory_ = hashes; }

    std::vector<Move> getLivePV();
    int getLiveDepth() const { return liveDepth_.load(std::memory_order_relaxed); }
    int getLiveEval()  const { return liveEval_.load(std::memory_order_relaxed); }

    // Top N root moves from the last completed depth (for multi-PV display)
    struct RootMove { Move move; int score = 0; std::vector<Move> pv; };
    std::vector<RootMove> getTopRootMoves(int n = 3) const {
        std::lock_guard<std::mutex> lock(topRootMtx_);
        int cnt = std::min(n, (int)topRootMoves_.size());
        return std::vector<RootMove>(topRootMoves_.begin(), topRootMoves_.begin() + cnt);
    }

    void setNNUE(const NNUE::Network* net) { nnue_ = const_cast<NNUE::Network*>(net); }
    NNUE::Network* getNNUE() const { return nnue_; }

    void setDuckNNUE(DuckNNUE::Network* net) { duckNnue_ = net; }
    DuckNNUE::Network* getDuckNNUE() const { return duckNnue_; }

    /// Callback: depth, score_cp, nodes, nps, elapsed_ms, pv_string, pv_index
    std::function<void(int, int, uint64_t, uint64_t, int64_t, const std::string&, int)> onInfoCallback;

    static uint64_t computeHash(const Board& board);
    static void     initZobrist();

    // ----------------------------------------------------------------
    // Public constants
    // ----------------------------------------------------------------
    static constexpr int    MATE_SCORE       = 100000;
    static constexpr size_t TT_ENTRY_BYTES   = 16;
    static constexpr size_t SELFPLAY_TT_SIZE = 1 << 20;

    // ----------------------------------------------------------------
    // Optional features
    // ----------------------------------------------------------------
    void resizeTT(size_t entries)     { tt_.resize(entries); }
    void setThreadCount(int n)        { numThreads_ = std::max(1, n); }
    void setMultiPV(int n)            { multiPV_ = n; }
    void setContempt(int cp)          { contempt_ = cp; }
    void clearSearchState();
    void startPonder()                {}
    void ponderHit()                  {}
    void ponderHit(int softMs, int hardMs) { setTimeLimits(softMs, hardMs); }
    uint64_t getCumulativeNodes() const { return cumulativeNodes_; }
    int      getMultiPV()     const   { return multiPV_; }
    uint64_t getTBHits()      const   { return tbHits_; }
    Move     getPonderMove()  const   { return ponderMove_; }

    /// Lazy SMP: point this engine at an external shared TT (used by helper threads)
    struct TTEntry {
        uint64_t key   = 0;    // 8 bytes
        int32_t  score = 0;    // 4 bytes
        int16_t  depth = -1;   // 2 bytes
        uint8_t  flag  = 2;    // 1 byte  (TT_UPPER=2)
        uint8_t  gen   = 0;    // 1 byte
        uint16_t best  = 0;    // 2 bytes (packed move via packMove/unpackMove)
        // Total: 18 bytes (vs ~44 with Move best{})
        // Padding to 20 bytes — still 3x smaller than before, much better cache utilization
    };
    void setSharedTT(std::vector<TTEntry>* sharedTT) { sharedTT_ = sharedTT; }

private:
    /* ---------- constants ---------- */
    static constexpr int INF     = MATE_SCORE + 1;
    static constexpr int MAX_PLY = 64;
    static constexpr size_t TT_SIZE = 1 << 22;

    /* ---------- optional feature state ---------- */
    int      multiPV_    = 1;
    int      contempt_   = 0;
    uint64_t tbHits_     = 0;
    Move     ponderMove_ = {};
    int      numThreads_ = 1;   // Lazy SMP thread count

    /* ---------- search ---------- */
    int search(Board& board, int depth, int alpha, int beta,
               int ply, bool doNull, bool isPV);
    int qsearch(Board& board, int alpha, int beta, int ply);

    /* ---------- evaluation ---------- */
    int evaluate(const Board& board);
    int evaluateWithAcc(const Board& board, int ply);  // uses accStack_[ply] if valid

    /* ---------- NNUE ---------- */
    NNUE::Network*     nnue_     = nullptr;
    DuckNNUE::Network* duckNnue_ = nullptr;

    /* ---------- SEE ---------- */
    static int see(const Board& board, const Move& m);
    static PieceType findLVA(const Board& board, Square target, Color side,
                             const bool removed[8][8], Square& outSq);

    /* ---------- move ordering ---------- */
    void orderMoves(MoveList& moves, const Board& board,
                    int ply, const Move& hashMove) const;
    int  scoreMove(const Move& m, const Board& board,
                   int ply, const Move& hashMove) const;
    static int  mvvLva(const Board& board, const Move& m);
    static bool isCapture(const Board& board, const Move& m);

    /* ---------- Zobrist hashing ---------- */
    static bool     zobristReady_;
    static uint64_t zPiece_[2][7][64];
    static uint64_t zCastle_[16];
    static uint64_t zEP_[8];
    static uint64_t zSide_;
    static uint64_t zDuck_[64];

    /* ---------- Duck Chess search ---------- */
    int searchDuck(Board& board, int depth, int alpha, int beta, int ply,
                   DuckNNUE::QAccumulator* accStack,
                   DuckNNUE::QAccumulator* parentAcc = nullptr);
    int scoreDuckPlacement(const Board& board, Square duckSq, Color myColor) const;
    void orderDuckPlacements(SquareList& placements, const Board& board, Color myColor, int topN = 0) const;

    // Accumulator stack for duck chess incremental updates
    // Slot [ply] = post-duck; slot [MAX_PLY+ply] = post-chess scratch
    std::unique_ptr<DuckNNUE::QAccumulator[]> duckAccStack_;

    // ---- Standard chess incremental NNUE accumulator ----
    // accStack_[ply] holds the QAccumulator for the position at that ply.
    // Seeded at ply=0 (root) and propagated via fusedCopyAndUpdateQ.
    // FinnyTable handles king-move refreshes efficiently.
    NNUE::QAccumulator  accStack_[MAX_PLY + 4];
    NNUE::FinnyTable    finny_;

    /* ---------- Duck placement cache ---------- */
    // Small separate table: (post-chess board hash, duck-stripped) → best duck square + score.
    // Allows reusing the duck placement loop result when the same post-chess position recurs.
    // 64K entries × 12 bytes = 768KB — fits comfortably in L3.
    static constexpr size_t DUCK_TT_SIZE = 1 << 16;  // 64K entries
    struct DuckTTEntry {
        uint64_t key    = 0;   // post-chess hash with duck XOR'd out
        int16_t  score  = 0;   // best score found across all duck placements
        uint8_t  bestSq = 255; // best duck square (0-63, 255 = invalid)
        uint8_t  depth  = 0;   // depth at which this was stored
        uint8_t  gen    = 0;   // generation for replacement
        uint8_t  flag   = 2;   // TT_EXACT / TT_LOWER / TT_UPPER
        // 14 bytes total, padded to 16
        uint16_t _pad   = 0;
    };
    std::vector<DuckTTEntry> duckTT_;
    enum TTFlag : uint8_t { TT_EXACT, TT_LOWER, TT_UPPER };
    // TTEntry is public (defined above)
    std::vector<TTEntry> tt_;
    std::vector<TTEntry>* sharedTT_ = nullptr;  // Lazy SMP: shared with helper threads
    uint8_t ttGen_ = 0;

    // Helper: get the active TT (shared if set, own otherwise)
    std::vector<TTEntry>& activeTT() { return sharedTT_ ? *sharedTT_ : tt_; }

    /* ---------- transposition table ---------- */
    Move killers_[MAX_PLY][2]{};
    int  history_[2][64][64]{};
    Move countermoves_[2][64][64]{};
    Move previousMove_{};

    // 1-ply continuation history: indexed by [prev_piece_type-1][prev_to_sq][curr_piece_type-1][curr_to_sq]
    // Piece types 1-6 (Pawn=1..King=6), subtract 1 for 0-based indexing.
    // Reduced from [7][64][7][64] (~784KB) to [6][64][6][64] (~576KB) — fits better in L2/L3.
    int  contHist_[6][64][6][64]{};

    // Search stack: stores the move played at each ply for continuation history lookup
    Move moveStack_[MAX_PLY]{};
    int  pieceStack_[MAX_PLY]{};  // piece type (int) of the move played at each ply

    /* ---------- principal variation ---------- */
    Move pvTable_[MAX_PLY][MAX_PLY]{};
    int  pvLength_[MAX_PLY]{};
    std::vector<Move> lastPV_;
    int lastDepth_ = 0;

    /* ---------- live PV / depth / eval ---------- */
    std::mutex livePVMutex_;
    std::vector<Move> livePV_;
    std::atomic<int> liveDepth_{0};
    std::atomic<int> liveEval_{0};

    /* ---------- top root moves (multi-PV display) ---------- */
    mutable std::mutex       topRootMtx_;
    std::vector<RootMove>    topRootMoves_;

    /* ---------- repetition detection ---------- */
    std::vector<uint64_t> gameHistory_;
    uint64_t searchStack_[MAX_PLY]{};

    /* ---------- dynamic draw scoring ---------- */
    int rootEval_ = 0;
    int drawScore() const;

    /* ---------- search control ---------- */
    std::atomic<bool> stop_{false};
    uint64_t nodes_          = 0;
    uint64_t cumulativeNodes_ = 0;
    int softLimitMs_ = 3000;
    int hardLimitMs_ = 3000;
    std::chrono::steady_clock::time_point searchStart_;

    bool shouldStop();
    std::string pvToUCI(const std::vector<Move>& pv) const;
    int64_t elapsedMs() const;
};
