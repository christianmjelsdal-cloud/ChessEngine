#pragma once
#include "Board.h"
#include "Bitboard.h"
#include "Types.h"
#include <array>
#include <memory>
#include <vector>
#include <string>
#include <cmath>

namespace NNUE {
    // HalfKAv2: 64 king buckets × 10 non-king piece types × 64 squares = 40,960
    constexpr int NUM_FEATURES = 40960;
    constexpr int NUM_FEATURES_768 = 768;  // legacy encoding for data files
    constexpr int KING_BUCKETS = 64;
    constexpr int PIECES_PER_PERSPECTIVE = 10;  // 5 own + 5 opponent non-king types
    constexpr int L1_SIZE = 512;   // per-perspective accumulator size (optimized for CPU-only training)
    constexpr int L2_SIZE = 128;
    constexpr int L3_SIZE = 64;
    constexpr int WDL_SIZE = 3;

    // INT16 quantization scale for L1 feature transform
    constexpr int QA = 256;  // accumulator quantization scale

    // INT8 quantization scales for L2/L3 inference
    constexpr int QA_ACT = 127;   // activation quantization (SCReLU ∈ [0,1] → uint8 [0,127])
    constexpr int QW_L2 = 64;    // L2 weight quantization scale
    constexpr int QW_L3 = 64;    // L3 weight quantization scale

    // HalfKAv2 feature index computation
    // perspective: 0=White, 1=Black
    // For the given perspective:
    //   Own non-king pieces: pawn=0, knight=1, bishop=2, rook=3, queen=4
    //   Opponent non-king pieces: pawn=5, knight=6, bishop=7, rook=8, queen=9
    // king_sq: the perspective's own king square (0-63, as rank*8+col)
    // piece_sq: the piece's square (0-63)
    // For black perspective, squares are vertically mirrored before indexing
    int halfKAv2Feature(int king_sq, PieceType pt, Color pieceColor, int piece_sq, Color perspective);
    
    // Convert board square to 0-63 index (rank*8 + col)
    inline int squareIndex(int rank, int col) { return rank * 8 + col; }
    
    // Mirror a square vertically (flip rank)
    inline int mirrorSquare(int sq) { return sq ^ 56; }

    // Legacy feature index (768-encoding) for data file compatibility
    int featureIndex768(PieceType pt, Color pc, int rank, int col);

    // Alias used by tests and external code (maps to 768-encoding)
    inline int featureIndex(PieceType pt, Color pc, int rank, int col) {
        return featureIndex768(pt, pc, rank, col);
    }

    // Mirror a 768-encoded feature index by flipping the color bit (involution)
    inline int mirrorFeature(int i) { return i ^ 384; }

    // INFO [5.13]: alignas(64) wastes ~64B padding per instance (128 instances = ~8KB).
    // Consider packing tail fields into the aligned region.
    struct alignas(64) Accumulator {
        alignas(64) std::array<float, L1_SIZE> white{};
        alignas(64) std::array<float, L1_SIZE> black{};
        bool valid = false;
        int whiteKingSq = -1;  // cached king squares for incremental updates
        int blackKingSq = -1;
    };

    // PHASE 3: Quantized INT16 accumulator for fast incremental updates (2x throughput vs float with SIMD)
    struct alignas(64) QAccumulator {
        alignas(64) std::array<int16_t, L1_SIZE> white{};
        alignas(64) std::array<int16_t, L1_SIZE> black{};
        bool valid = false;
        int whiteKingSq = -1;
        int blackKingSq = -1;
    };

    // Finny Table: caches last-refreshed accumulator per king bucket
    // Enables O(delta) refresh on king moves instead of O(all_pieces) rebuild
    struct FinnyEntry {
        alignas(64) std::array<int16_t, L1_SIZE> values{};
        // Piece placement snapshot at cache time (non-king pieces only)
        // [color 0=White,1=Black][pieceType 1=Pawn..5=Queen]
        Bitboard pieces[2][6] = {};
        bool valid = false;
    };

    struct FinnyTable {
        // [perspective 0=White,1=Black][kingSq 0-63]
        FinnyEntry entries[2][64] = {};
        void clear() {
            for (auto& persp : entries)
                for (auto& e : persp)
                    e.valid = false;
        }
    };

    // Phase head: produces WDL (win/draw/loss) logits from L3 output
    struct PhaseHead {
        // weights[j][i]: j=output(0=win,1=draw,2=loss), i=input(0..L3_SIZE-1)
        alignas(64) std::array<std::array<float, L3_SIZE>, WDL_SIZE> weights{};
        std::array<float, WDL_SIZE> biases{};
    };

    // WARNING: Network is very large (~80+ MB) due to HalfKAv2 weight arrays.
    // MUST be heap-allocated (e.g., std::unique_ptr<Network>).
    // alignas(64) ensures cache-line alignment for SIMD.
    class alignas(64) Network {
    public:
        Network();

        // Full evaluation (recomputes accumulator from scratch)
        // AUDIT FIX 14: This method is thread-safe for concurrent reads —
        // it only reads weight arrays and writes to stack-local accumulators.
        int evaluate(const Board& board) const;

        // Incremental accumulator management
        void refreshAccumulator(const Board& board, Accumulator& acc) const;
        void incrementalUpdate(const Board& board, Accumulator& acc,
                              int fromRank, int fromCol, int toRank, int toCol,
                              PieceType movedPiece, Color movedColor,
                              PieceType capturedPiece = PieceType::None, 
                              Color capturedColor = Color::White) const;
        int forward(const Accumulator& acc, Color sideToMove, float phase) const;
        int forward(const Accumulator& acc, Color sideToMove) const;

        // Compute material phase from board (0.0=endgame, 1.0=opening)
        static float computePhase(const Board& board);

        // Incremental feature add/remove on an existing accumulator
        // featureIdx must be in [0, NUM_FEATURES)
        // Perspective-aware: separate feature index per side (correct for HalfKAv2)
        void addFeature(int whiteFeatureIdx, int blackFeatureIdx, Accumulator& acc) const;
        void removeFeature(int whiteFeatureIdx, int blackFeatureIdx, Accumulator& acc) const;
        // Legacy single-index (same weights to both perspectives — use after full refresh only)
        void addFeature(int featureIdx, Accumulator& acc) const;
        void removeFeature(int featureIdx, Accumulator& acc) const;

        // Weight management
        bool loadWeights(const std::string& filename);
        bool saveWeights(const std::string& filename);
        void randomizeWeights(float scale = 1.0f, int seed = 42);

        // Weights (public for training access)
        // L1: HalfKAv2 feature transform (per-perspective, shared weights)
        // NOTE: This is ~80 MB — stored on the heap via unique_ptr to avoid
        // blowing the MSVC compiler heap (C1060) at compile time.
        // Allocated in the constructor via make_unique.
        std::unique_ptr<std::array<std::array<float, L1_SIZE>, NUM_FEATURES>> L1_weights;
        alignas(64) std::array<float, L1_SIZE> L1_biases{};

        // Quantized L1 weights and biases (INT16, scale = QA)
        std::unique_ptr<std::array<std::array<int16_t, L1_SIZE>, NUM_FEATURES>> L1_weights_q;
        alignas(64) std::array<int16_t, L1_SIZE> L1_biases_q{};

        // L2: 1024 -> 128 (512 per perspective, concatenated)
        alignas(64) std::array<std::array<float, L2_SIZE>, L1_SIZE * 2> L2_weights{};
        alignas(64) std::array<float, L2_SIZE> L2_biases{};

        // L3: 64 -> 32
        alignas(64) std::array<std::array<float, L3_SIZE>, L2_SIZE> L3_weights{};
        alignas(64) std::array<float, L3_SIZE> L3_biases{};

        // Phase heads: 3 heads (opening, middlegame, endgame) each producing WDL
        PhaseHead head_opening{};
        PhaseHead head_middlegame{};
        PhaseHead head_endgame{};

        // Legacy scalar output layer (used by NNUETrainer — maps L3 → single value)
        alignas(64) std::array<float, L3_SIZE> output_weights{};
        float output_bias = 0.0f;

        // Transposed weight copies for cache-friendly SIMD inference
        alignas(64) std::array<std::array<float, L1_SIZE * 2>, L2_SIZE> L2_weights_T{};
        alignas(64) std::array<std::array<float, L2_SIZE>, L3_SIZE>     L3_weights_T{};

        // INT8 quantized transposed weights for L2/L3 (OPT A: 4× throughput vs float SIMD)
        alignas(64) std::array<std::array<int8_t, L1_SIZE * 2>, L2_SIZE> L2_weights_T_q{};
        alignas(64) std::array<int32_t, L2_SIZE> L2_biases_q{};   // pre-scaled: round(bias * QA_ACT * QW_L2)
        alignas(64) std::array<std::array<int8_t, L2_SIZE>, L3_SIZE> L3_weights_T_q{};
        alignas(64) std::array<int32_t, L3_SIZE> L3_biases_q{};   // pre-scaled: round(bias * QA_ACT * QW_L3)

        void transposeWeights();

        // PHASE 3: Quantized evaluation (uses INT16 accumulator for L1, float for L2+)
        int evaluateQ(const Board& board) const;
        
        // PHASE 3: Quantized accumulator management — SSE2/AVX2 int16 arithmetic
        void refreshAccumulatorQ(const Board& board, QAccumulator& acc) const;
        // OPT #1: Finny-table-accelerated refresh — delta from cached bucket state
        void refreshAccumulatorQFinny(const Board& board, QAccumulator& acc,
                                       FinnyTable& finny) const;
        void incrementalUpdateQ(const Board& board, QAccumulator& acc,
                               int fromRank, int fromCol, int toRank, int toCol,
                               PieceType movedPiece, Color movedColor,
                               PieceType capturedPiece = PieceType::None,
                               Color capturedColor = Color::White) const;

        /// TIER-1 FIX #5: Fused copy+update — reads parent acc, applies delta,
        /// writes to child acc in a single pass (2 memory passes instead of 3).
        /// Eliminates the separate 4KB memcpy from parent->child.
        void fusedCopyAndUpdateQ(const Board& board,
                                 const QAccumulator& parent, QAccumulator& child,
                                 int fromRank, int fromCol, int toRank, int toCol,
                                 PieceType movedPiece, Color movedColor,
                                 PieceType capturedPiece, Color capturedColor) const;
        // PHASE 3: Forward pass dequantizes int16→float at L1 output, then SCReLU + L2/L3
        int forwardQ(const QAccumulator& acc, Color sideToMove, float phase) const;
        int forwardQ(const QAccumulator& acc, Color sideToMove) const;
        
        // PHASE 3: Quantize float weights to INT16 (call after loading or randomizing weights)
        void quantizeWeights();
        void quantizeL2L3Weights();  // OPT A: quantize L2/L3 to INT8 after loading
        void releaseFloatWeights();  // free ~160 MB float L1 after quantization
    };
}
