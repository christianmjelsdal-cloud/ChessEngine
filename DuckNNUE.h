#pragma once
#include "Board.h"
#include "Types.h"
#include "NNUE.h"
#include <array>
#include <vector>
#include <string>
#include <cmath>

namespace DuckNNUE {
    // 832 features = 768 standard piece features + 64 duck-square features
    constexpr int NUM_FEATURES = 832;
    constexpr int DUCK_FEATURE_OFFSET = 768;  // features 768-831 encode duck square
    constexpr int L1_SIZE = NNUE::L1_SIZE;    // same hidden layer sizes as standard
    constexpr int L2_SIZE = NNUE::L2_SIZE;
    constexpr int L3_SIZE = NNUE::L3_SIZE;

    // Duck feature index: duck on square (rank, col) -> feature 768 + rank*8 + col
    inline int duckFeatureIndex(int rank, int col) {
        return DUCK_FEATURE_OFFSET + rank * 8 + col;
    }

    // Mirror a duck feature: flip rank (7 - rank), keep col
    inline int mirrorDuckFeature(int feature) {
        if (feature < DUCK_FEATURE_OFFSET) {
            return NNUE::mirrorFeature(feature);  // standard piece feature
        }
        int sq = feature - DUCK_FEATURE_OFFSET;
        int rank = sq / 8;
        int col = sq % 8;
        int mirroredRank = 7 - rank;
        return DUCK_FEATURE_OFFSET + mirroredRank * 8 + col;
    }

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
        // L1: per-perspective feature transform (832 input features)
        alignas(16) std::array<std::array<float, L1_SIZE>, NUM_FEATURES> L1_weights{};
        alignas(16) std::array<float, L1_SIZE> L1_biases{};

        // L2: (L1_SIZE*2) -> L2_SIZE
        alignas(16) std::array<std::array<float, L2_SIZE>, L1_SIZE * 2> L2_weights{};
        alignas(16) std::array<float, L2_SIZE> L2_biases{};

        // L3: L2_SIZE -> L3_SIZE
        alignas(16) std::array<std::array<float, L3_SIZE>, L2_SIZE> L3_weights{};
        alignas(16) std::array<float, L3_SIZE> L3_biases{};

        // Output: L3_SIZE -> 1
        alignas(16) std::array<float, L3_SIZE> output_weights{};
        float output_bias = 0.0f;
    };
}
