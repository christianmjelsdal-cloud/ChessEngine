#pragma once
#include "Board.h"
#include "Types.h"
#include <array>
#include <vector>
#include <string>
#include <cmath>

namespace NNUE {
    constexpr int NUM_FEATURES = 768;
    constexpr int L1_SIZE = 512;  // CHANGED: 256 -> 512 for 42M+ position dataset
    constexpr int L2_SIZE = 128;  // CHANGED: 64 -> 128 for more capacity
    constexpr int L3_SIZE = 128;  // CHANGED: 64 -> 128 for more capacity

    // Feature index helpers
    int featureIndex(PieceType pt, Color pc, int rank, int col);
    int mirrorFeature(int feature);  // flip for opponent's perspective

    struct Accumulator {
        alignas(16) std::array<float, L1_SIZE> white{};
        alignas(16) std::array<float, L1_SIZE> black{};
        bool valid = false;
    };

    class Network {
    public:
        Network();

        // Full evaluation (recomputes accumulator from scratch)
        int evaluate(const Board& board);

        // Incremental accumulator management
        void refreshAccumulator(const Board& board, Accumulator& acc);
        void addFeature(int feature, Accumulator& acc);
        void removeFeature(int feature, Accumulator& acc);
        int forward(const Accumulator& acc, Color sideToMove);

        // Weight management
        bool loadWeights(const std::string& filename);
        bool saveWeights(const std::string& filename);
        void randomizeWeights(float scale = 0.1f);

        // Weights (public for training access)
        // L1: per-perspective feature transform
        // Aligned for SIMD access
        alignas(16) std::array<std::array<float, L1_SIZE>, NUM_FEATURES> L1_weights{};
        alignas(16) std::array<float, L1_SIZE> L1_biases{};

        // L2: 1024 -> 128 (CHANGED from 512->64)
        alignas(16) std::array<std::array<float, L2_SIZE>, L1_SIZE * 2> L2_weights{};
        alignas(16) std::array<float, L2_SIZE> L2_biases{};

        // L3: 64 -> 64 (CHANGED from 32)
        alignas(16) std::array<std::array<float, L3_SIZE>, L2_SIZE> L3_weights{};
        alignas(16) std::array<float, L3_SIZE> L3_biases{};

        // Output: 64 -> 1 (CHANGED from 32)
        alignas(16) std::array<float, L3_SIZE> output_weights{};
        float output_bias = 0.0f;
    };
}
