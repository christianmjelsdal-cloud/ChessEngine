#pragma once
#include "Board.h"
#include "Types.h"
#include "NNUE.h"
#include <array>
#include <vector>
#include <string>
#include <cmath>
#include <memory>

namespace DuckNNUE {
    // 832 features = 768 standard piece features + 64 duck-square features
    constexpr int NUM_FEATURES = 832;
    constexpr int DUCK_FEATURE_OFFSET = 768;
    constexpr int L1_SIZE = NNUE::L1_SIZE;
    constexpr int L2_SIZE = NNUE::L2_SIZE;
    constexpr int L3_SIZE = NNUE::L3_SIZE;

    // Quantization constants (same as standard NNUE)
    constexpr int QA     = NNUE::QA;
    constexpr int QA_ACT = NNUE::QA_ACT;
    constexpr int QW_L2  = NNUE::QW_L2;
    constexpr int QW_L3  = NNUE::QW_L3;

    inline int duckFeatureIndex(int rank, int col) {
        return DUCK_FEATURE_OFFSET + rank * 8 + col;
    }

    inline int mirrorDuckFeature(int feature) {
        if (feature < DUCK_FEATURE_OFFSET)
            return NNUE::mirrorFeature(feature);
        int sq = feature - DUCK_FEATURE_OFFSET;
        int rank = sq / 8, col = sq % 8;
        return DUCK_FEATURE_OFFSET + (7 - rank) * 8 + col;
    }

    // Float accumulator (used by training)
    struct Accumulator {
        alignas(16) std::array<float, L1_SIZE> white{};
        alignas(16) std::array<float, L1_SIZE> black{};
        bool valid = false;
    };

    // INT16 quantized accumulator (used by evaluateQ)
    struct QAccumulator {
        std::array<int16_t, L1_SIZE> white{};
        std::array<int16_t, L1_SIZE> black{};
        bool valid = false;
    };

    class Network {
    public:
        Network();

        // Float evaluation (used by training code)
        int evaluate(const Board& board) const;
        void refreshAccumulator(const Board& board, Accumulator& acc) const;
        void addFeature(int feature, Accumulator& acc) const;
        void removeFeature(int feature, Accumulator& acc) const;
        int forward(const Accumulator& acc, Color sideToMove) const;

        // INT16 quantized evaluation (fast inference path)
        int evaluateQ(const Board& board) const;
        void refreshAccumulatorQ(const Board& board, QAccumulator& acc) const;
        void addFeatureQ(int feature, QAccumulator& acc) const;
        void removeFeatureQ(int feature, QAccumulator& acc) const;
        int forwardQ(const QAccumulator& acc, Color sideToMove) const;

        // Weight management
        bool loadWeights(const std::string& filename);
        bool saveWeights(const std::string& filename);
        void randomizeWeights(float scale = 0.1f);
        void transposeWeights();
        void quantizeWeights();   // populate *_q arrays from float weights

        // Float weights (public for training)
        alignas(16) std::array<std::array<float, L1_SIZE>, NUM_FEATURES> L1_weights{};
        alignas(16) std::array<float, L1_SIZE> L1_biases{};
        alignas(16) std::array<std::array<float, L2_SIZE>, L1_SIZE * 2> L2_weights{};
        alignas(16) std::array<float, L2_SIZE> L2_biases{};
        alignas(16) std::array<std::array<float, L3_SIZE>, L2_SIZE> L3_weights{};
        alignas(16) std::array<float, L3_SIZE> L3_biases{};
        alignas(16) std::array<float, L3_SIZE> output_weights{};
        float output_bias = 0.0f;

        // Transposed float caches
        alignas(16) std::array<std::array<float, L1_SIZE * 2>, L2_SIZE> L2_weights_T{};
        alignas(16) std::array<std::array<float, L2_SIZE>, L3_SIZE>     L3_weights_T{};

        // INT16 quantized L1 (heap — 832 * 512 * 2 = 832KB)
        std::unique_ptr<std::array<std::array<int16_t, L1_SIZE>, NUM_FEATURES>> L1_weights_q;
        alignas(32) std::array<int16_t, L1_SIZE> L1_biases_q{};

        // INT8 quantized L2/L3 transposed
        alignas(32) std::array<std::array<int8_t, L1_SIZE * 2>, L2_SIZE> L2_weights_T_q{};
        alignas(32) std::array<int32_t, L2_SIZE> L2_biases_q{};
        alignas(32) std::array<std::array<int8_t, L2_SIZE>, L3_SIZE> L3_weights_T_q{};
        alignas(32) std::array<int32_t, L3_SIZE> L3_biases_q{};
    };
}

