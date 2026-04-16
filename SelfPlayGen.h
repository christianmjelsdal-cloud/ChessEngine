#pragma once
#include "AssetPath.h"
#include "Board.h"
#include "Engine.h"
#include "NNUE.h"
#include "DuckNNUE.h"
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

// =============================================================================
//  SelfPlayGen - native C++ self-play data generator
//
//  Plays the engine against itself and writes positions to a binary file
//  fully compatible with train_nnue.py.
//
//  Binary format (little-endian, matches generate_selfplay.py exactly):
//    Header:  [char[4]   magic = "NNUE"]
//             [uint8_t   version = 1]
//             [uint32_t  position_count]
//    Records: [uint16_t  num_features]
//             [uint16_t  feature_indices[num_features]]  (sorted ascending)
//             [uint8_t   stm]     0 = white to move, 1 = black to move
//             [float32   result]  1.0 = white wins, 0.5 = draw, 0.0 = black wins
//             [float32   eval]    White's perspective eval in centipawns
//
//  Feature encoding (matches generate_selfplay.py):
//    index = piece_type_offset * 64 + rank * 8 + col
//    White offsets: Pawn=0, Knight=1, Bishop=2, Rook=3, Queen=4, King=5
//    Black offsets: Pawn=6, Knight=7, Bishop=8, Rook=9, Queen=10, King=11
//    (rank 0 = white's back rank, col 0 = a-file - matches python-chess squares)
// =============================================================================

class SelfPlayGen {
public:
    enum class GameResult { WhiteWins, BlackWins, Draw };

    /// Termination reason — why a game ended.
    enum class TermReason {
        Checkmate,    ///< one side is in checkmate
        Stalemate,    ///< no legal moves, not in check
        FiftyMove,    ///< 50-move rule (halfMoveClock >= 100)
        Threefold,    ///< threefold repetition
        Resign,       ///< resign adjudication (eval beyond threshold for N plies)
        DrawAdj,      ///< draw adjudication (eval within threshold for N plies)
        DeadDraw,     ///< dead-draw adjudication (stricter, near-zero eval)
        MaxPlies,     ///< game reached maximum ply limit
        Timeout       ///< per-game wall-clock timeout
    };

    /// Full outcome returned by playGame — includes reason, final ply, and eval.
    struct GameOutcome {
        GameResult result   = GameResult::Draw;
        TermReason reason   = TermReason::MaxPlies;
        int        finalPly = 0;      ///< ply at which the game ended
        float      finalEval = 0.0f;  ///< last search eval (White's POV, centipawns)
    };

    struct Config {
        int         games        = 1000;  ///< total games to play
        int         moveTimeMs   = 0;     ///< milliseconds per move (0 = use searchDepth instead)
        int         searchDepth  = 5;     ///< depth-based search (0 = use moveTimeMs instead)
        int         workers      = 1;     ///< parallel threads (each with own Engine)
        int         maxPlies     = 250;   ///< max half-moves per game before draw (lowered from 300)
        int         gameTimeLimitSec = 120; ///< per-game wall-clock timeout (seconds, 0 = unlimited)
        int         resignCp     = 500;   ///< resign threshold (centipawns, White's perspective)
        int         resignCount  = 3;     ///< consecutive plies above threshold to trigger resign
        int         drawCp       = 8;     ///< draw threshold (centipawns) — lowered from 15 for fewer premature draws
        int         drawCount    = 6;     ///< consecutive plies below threshold to trigger draw (lowered from 8 for faster adjudication)
        int         drawMinPly   = 40;    ///< earliest ply for draw adjudication (lowered from 60 for faster adjudication)
        int         drawAdjMoves     = 12;  ///< plies of near-zero eval before adjudicating draw — raised from 10
        int         drawAdjThreshold = 4;   ///< centipawn threshold for "dead equal" — tightened from 5
        int         drawAdjMinMove   = 50;  ///< minimum move number before adjudication — raised from 40
        int         openingPlies = 4;     ///< plies using softmax move selection for opening diversity
        std::string outputPath   = assetPath("assets/selfplay.bin");
        std::string openingsFile;         ///< path to opening book (one FEN per line)

        // --- Phase 1A: Softmax move selection for diversity ---
        float       openingTemp  = 1.5f;  ///< softmax temperature for opening plies (0 = best move, higher = more random)
        int         softmaxPlies = 8;     ///< post-opening plies using softmax move selection
        float       softmaxTemp  = 0.5f;  ///< softmax temperature for post-opening phase

        // --- Contempt: penalize draws for more decisive games ---
        int         contemptCp   = 25;     ///< fixed contempt in centipawns (25cp default for decisive self-play games)

        // --- Phase 2B: Root move noise for game diversity ---
        float       rootNoiseEps = 0.0f;  ///< probability of replacing best move with a weighted alternative (0 = off, 0.05-0.15 recommended)

        // --- Phase 4B: Chess960 (FRC) starting positions ---
        float       frcMix       = 0.0f;  ///< fraction of games using random Chess960 starts (0.0-1.0)

        // --- Phase 3A: Position filtering for training data quality ---
        int         recordMinPly = 10;    ///< don't record positions before this ply (skip "book" territory)
        int         recordMaxEval = 2500; ///< don't record positions with |eval| > this (already decided)

        // --- Mixed depth strategy for throughput ---
        float       mixedDepthRatio = 0.0f; ///< fraction of games at reduced depth (0.0 = off, 0.8 = 80% at low depth)
        int         mixedDepthLow   = 4;    ///< depth for "fast" games when mixedDepthRatio > 0

        // --- Depth shuffle: sample from a distribution of depths ---
        bool        depthShuffle    = false; ///< when true, shuffled games sample depth from [mixedDepthLow, searchDepth) with geometric weighting
        float       depthShuffleBias = 2.0f; ///< geometric weight: P(d) ∝ bias^(d - low). Higher = more high-depth games

        // --- Optional logging callback (avoids dependency on TR_Globals) ---
        std::function<void(const std::string&)> logFn;  ///< if set, config dump and per-game lines are sent here

        // --- NPS sampling: emit NPS_SAMPLE lines at evenly-spaced game intervals ---
        int         npsSamples   = 0;  ///< number of NPS samples to emit (0 = disabled, use epochsPerGen)

        // --- Variant ---
        bool isDuckChess = false;  ///< generate duck chess positions (sets board.isDuckChess, uses 832-feature encoding)
    };

    /// Construct with a loaded NNUE network (pass nullptr for handcrafted eval).
    explicit SelfPlayGen(const NNUE::Network* nnue, const DuckNNUE::Network* duckNnue = nullptr);

    /// Run all games and write the output file.
    /// Prints progress every 10 games.
    /// Returns total positions written, or -1 on error.
    int generate(const Config& cfg);

    /// Encode a board into sorted NNUE feature indices (public for testing).
    static std::vector<uint16_t> boardToFeatures(const Board& board);

    /// Generate a random Chess960 starting position FEN.
    /// index 0-959 selects a specific arrangement; -1 = random.
    static std::string generateFRC960FEN(int index = -1);

private:
    struct PositionRecord {
        std::vector<uint16_t> features;
        uint8_t stm   = 0;
        float   eval   = 0.0f;    ///< White's perspective, in centipawns
        float   result = 0.0f;    ///< filled in after game ends
    };

    const NNUE::Network*     nnue_;      // standard NNUE (nullptr = handcrafted eval)
    const DuckNNUE::Network* duckNnue_; // duck NNUE (nullptr = handcrafted eval for duck chess)
    std::vector<std::string> m_openings;  ///< loaded opening FENs

    /// Play one game using the given engine instance.
    /// gameSeed seeds the RNG for opening randomisation.
    /// Appends PositionRecords (result field = 0, filled by caller).
    static GameOutcome playGame(Engine& engine,
                                const Config& cfg,
                                int gameSeed,
                                const std::vector<std::string>& openings,
                                std::vector<PositionRecord>& out,
                                int searchDepth);
};
