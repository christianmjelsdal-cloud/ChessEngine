#include "DuckNNUE.h"
#include <fstream>
#include <random>
#include <cmath>
#include <algorithm>
#include <cstring>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <immintrin.h>
#endif

namespace DuckNNUE {

    // SSE helper: SCReLU (Squared Clipped ReLU) on 4 floats: max(0, min(x, 1))^2
    static inline __m128 screlu_sse(__m128 x) {
        __m128 zero = _mm_setzero_ps();
        __m128 one = _mm_set1_ps(1.0f);
        __m128 clamped = _mm_max_ps(zero, _mm_min_ps(x, one));
        return _mm_mul_ps(clamped, clamped);
    }

    Network::Network() {
        randomizeWeights();
    }

    int Network::evaluate(const Board& board) {
        Accumulator acc;
        refreshAccumulator(board, acc);
        return forward(acc, board.turn);
    }

    void Network::refreshAccumulator(const Board& board, Accumulator& acc) {
        // Initialize with biases
        for (int j = 0; j < L1_SIZE; j += 4) {
            __m128 bias = _mm_loadu_ps(&L1_biases[j]);
            _mm_storeu_ps(&acc.white[j], bias);
            _mm_storeu_ps(&acc.black[j], bias);
        }

        // Standard piece features (0-767)
        for (int rank = 0; rank < 8; ++rank) {
            for (int col = 0; col < 8; ++col) {
                Piece piece = board.squares[rank][col];
                if (piece.isNone() || piece.isDuck()) continue;

                int wFeature = NNUE::featureIndex(piece.type, piece.color, rank, col);
                int bFeature = NNUE::mirrorFeature(wFeature);

                const float* wWeights = L1_weights[wFeature].data();
                float* wAcc = acc.white.data();
                for (int j = 0; j < L1_SIZE; j += 4) {
                    __m128 a = _mm_loadu_ps(&wAcc[j]);
                    __m128 w = _mm_loadu_ps(&wWeights[j]);
                    _mm_storeu_ps(&wAcc[j], _mm_add_ps(a, w));
                }

                const float* bWeights = L1_weights[bFeature].data();
                float* bAcc = acc.black.data();
                for (int j = 0; j < L1_SIZE; j += 4) {
                    __m128 a = _mm_loadu_ps(&bAcc[j]);
                    __m128 w = _mm_loadu_ps(&bWeights[j]);
                    _mm_storeu_ps(&bAcc[j], _mm_add_ps(a, w));
                }
            }
        }

        // Duck feature (768-831): encode duck position
        if (board.isDuckChess && board.duckSquare.isValid()) {
            int duckFeat = duckFeatureIndex(board.duckSquare.rank, board.duckSquare.col);
            // Duck is symmetric — same feature for both perspectives
            // (the duck blocks both sides equally, so white and black see the same duck)
            const float* dWeights = L1_weights[duckFeat].data();

            float* wAcc = acc.white.data();
            for (int j = 0; j < L1_SIZE; j += 4) {
                __m128 a = _mm_loadu_ps(&wAcc[j]);
                __m128 w = _mm_loadu_ps(&dWeights[j]);
                _mm_storeu_ps(&wAcc[j], _mm_add_ps(a, w));
            }

            // For black's perspective, mirror the duck feature (flip rank)
            int mirroredDuckFeat = mirrorDuckFeature(duckFeat);
            const float* dWeightsB = L1_weights[mirroredDuckFeat].data();

            float* bAcc = acc.black.data();
            for (int j = 0; j < L1_SIZE; j += 4) {
                __m128 a = _mm_loadu_ps(&bAcc[j]);
                __m128 w = _mm_loadu_ps(&dWeightsB[j]);
                _mm_storeu_ps(&bAcc[j], _mm_add_ps(a, w));
            }
        }

        acc.valid = true;
    }

    void Network::addFeature(int feature, Accumulator& acc) {
        int mirrored = mirrorDuckFeature(feature);  // handles both standard and duck features

        const float* wWeights = L1_weights[feature].data();
        float* wAcc = acc.white.data();
        for (int j = 0; j < L1_SIZE; j += 4) {
            __m128 a = _mm_loadu_ps(&wAcc[j]);
            __m128 w = _mm_loadu_ps(&wWeights[j]);
            _mm_storeu_ps(&wAcc[j], _mm_add_ps(a, w));
        }

        const float* bWeights = L1_weights[mirrored].data();
        float* bAcc = acc.black.data();
        for (int j = 0; j < L1_SIZE; j += 4) {
            __m128 a = _mm_loadu_ps(&bAcc[j]);
            __m128 w = _mm_loadu_ps(&bWeights[j]);
            _mm_storeu_ps(&bAcc[j], _mm_add_ps(a, w));
        }
    }

    void Network::removeFeature(int feature, Accumulator& acc) {
        int mirrored = mirrorDuckFeature(feature);

        const float* wWeights = L1_weights[feature].data();
        float* wAcc = acc.white.data();
        for (int j = 0; j < L1_SIZE; j += 4) {
            __m128 a = _mm_loadu_ps(&wAcc[j]);
            __m128 w = _mm_loadu_ps(&wWeights[j]);
            _mm_storeu_ps(&wAcc[j], _mm_sub_ps(a, w));
        }

        const float* bWeights = L1_weights[mirrored].data();
        float* bAcc = acc.black.data();
        for (int j = 0; j < L1_SIZE; j += 4) {
            __m128 a = _mm_loadu_ps(&bAcc[j]);
            __m128 w = _mm_loadu_ps(&bWeights[j]);
            _mm_storeu_ps(&bAcc[j], _mm_sub_ps(a, w));
        }
    }

    int Network::forward(const Accumulator& acc, Color sideToMove) {
        // Build input with SCReLU — identical to standard NNUE forward pass
        alignas(16) float input[L1_SIZE * 2];

        const auto& stmAcc = (sideToMove == Color::White) ? acc.white : acc.black;
        const auto& oppAcc = (sideToMove == Color::White) ? acc.black : acc.white;

        for (int i = 0; i < L1_SIZE; i += 4) {
            __m128 val = _mm_loadu_ps(&stmAcc[i]);
            _mm_storeu_ps(&input[i], screlu_sse(val));
        }
        for (int i = 0; i < L1_SIZE; i += 4) {
            __m128 val = _mm_loadu_ps(&oppAcc[i]);
            _mm_storeu_ps(&input[L1_SIZE + i], screlu_sse(val));
        }

        // L2
        alignas(16) float l2Out[L2_SIZE];
        for (int j = 0; j < L2_SIZE; ++j) {
            __m128 sum = _mm_setzero_ps();
            for (int i = 0; i < L1_SIZE * 2; i += 4) {
                __m128 inp = _mm_loadu_ps(&input[i]);
                __m128 w = _mm_set_ps(
                    L2_weights[i + 3][j],
                    L2_weights[i + 2][j],
                    L2_weights[i + 1][j],
                    L2_weights[i + 0][j]
                );
                sum = _mm_add_ps(sum, _mm_mul_ps(inp, w));
            }
            __m128 shuf = _mm_movehdup_ps(sum);
            __m128 sums = _mm_add_ps(sum, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            float dotProduct = _mm_cvtss_f32(sums);

            float val = dotProduct + L2_biases[j];
            float clamped = std::max(0.0f, std::min(val, 1.0f));
            l2Out[j] = clamped * clamped;
        }

        // L3
        alignas(16) float l3Out[L3_SIZE];
        for (int j = 0; j < L3_SIZE; ++j) {
            __m128 sum = _mm_setzero_ps();
            for (int i = 0; i < L2_SIZE; i += 4) {
                __m128 inp = _mm_loadu_ps(&l2Out[i]);
                __m128 w = _mm_set_ps(
                    L3_weights[i + 3][j],
                    L3_weights[i + 2][j],
                    L3_weights[i + 1][j],
                    L3_weights[i + 0][j]
                );
                sum = _mm_add_ps(sum, _mm_mul_ps(inp, w));
            }
            __m128 shuf = _mm_movehdup_ps(sum);
            __m128 sums = _mm_add_ps(sum, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            float dotProduct = _mm_cvtss_f32(sums);

            float val = dotProduct + L3_biases[j];
            float clamped = std::max(0.0f, std::min(val, 1.0f));
            l3Out[j] = clamped * clamped;
        }

        // Output
        __m128 outputSum = _mm_setzero_ps();
        for (int i = 0; i < L3_SIZE; i += 4) {
            __m128 inp = _mm_loadu_ps(&l3Out[i]);
            __m128 w = _mm_loadu_ps(&output_weights[i]);
            outputSum = _mm_add_ps(outputSum, _mm_mul_ps(inp, w));
        }
        __m128 shuf = _mm_movehdup_ps(outputSum);
        __m128 sums = _mm_add_ps(outputSum, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        float output = _mm_cvtss_f32(sums) + output_bias;

        return static_cast<int>(output * 400.0f);
    }

    bool Network::loadWeights(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) return false;

        file.read(reinterpret_cast<char*>(L1_weights.data()), sizeof(L1_weights));
        file.read(reinterpret_cast<char*>(L1_biases.data()), sizeof(L1_biases));
        file.read(reinterpret_cast<char*>(L2_weights.data()), sizeof(L2_weights));
        file.read(reinterpret_cast<char*>(L2_biases.data()), sizeof(L2_biases));
        file.read(reinterpret_cast<char*>(L3_weights.data()), sizeof(L3_weights));
        file.read(reinterpret_cast<char*>(L3_biases.data()), sizeof(L3_biases));
        file.read(reinterpret_cast<char*>(output_weights.data()), sizeof(output_weights));
        file.read(reinterpret_cast<char*>(&output_bias), sizeof(output_bias));

        return file.good();
    }

    bool Network::saveWeights(const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) return false;

        file.write(reinterpret_cast<const char*>(L1_weights.data()), sizeof(L1_weights));
        file.write(reinterpret_cast<const char*>(L1_biases.data()), sizeof(L1_biases));
        file.write(reinterpret_cast<const char*>(L2_weights.data()), sizeof(L2_weights));
        file.write(reinterpret_cast<const char*>(L2_biases.data()), sizeof(L2_biases));
        file.write(reinterpret_cast<const char*>(L3_weights.data()), sizeof(L3_weights));
        file.write(reinterpret_cast<const char*>(L3_biases.data()), sizeof(L3_biases));
        file.write(reinterpret_cast<const char*>(output_weights.data()), sizeof(output_weights));
        file.write(reinterpret_cast<const char*>(&output_bias), sizeof(output_bias));

        return file.good();
    }

    void Network::randomizeWeights(float scale) {
        std::mt19937 rng(42);

        {
            float stddev = scale * std::sqrt(2.0f / (NUM_FEATURES + L1_SIZE));
            std::normal_distribution<float> dist(0.0f, stddev);
            for (auto& row : L1_weights)
                for (auto& w : row)
                    w = dist(rng);
            L1_biases.fill(0.0f);
        }

        {
            float stddev = scale * std::sqrt(2.0f / (L1_SIZE * 2 + L2_SIZE));
            std::normal_distribution<float> dist(0.0f, stddev);
            for (auto& row : L2_weights)
                for (auto& w : row)
                    w = dist(rng);
            L2_biases.fill(0.0f);
        }

        {
            float stddev = scale * std::sqrt(2.0f / (L2_SIZE + L3_SIZE));
            std::normal_distribution<float> dist(0.0f, stddev);
            for (auto& row : L3_weights)
                for (auto& w : row)
                    w = dist(rng);
            L3_biases.fill(0.0f);
        }

        {
            float stddev = scale * std::sqrt(2.0f / (L3_SIZE + 1));
            std::normal_distribution<float> dist(0.0f, stddev);
            for (auto& w : output_weights)
                w = dist(rng);
            output_bias = 0.0f;
        }
    }

} // namespace DuckNNUE
