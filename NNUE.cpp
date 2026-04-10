#include "NNUE.h"
#include <fstream>
#include <random>
#include <cmath>
#include <algorithm>
#include <cstring>

// Improvement #7: SIMD vectorization
// MSVC x64 always supports SSE2. We use SSE for key inner loops.
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <immintrin.h>
#endif

namespace NNUE {

    // =========================================================================
    // SSE helper: SCReLU (Squared Clipped ReLU) on 4 floats: max(0, min(x, 1))^2
    // CHANGED: From ClippedReLU to SCReLU — modern NNUE standard (Stockfish uses this)
    // SCReLU preserves more gradient information and improves training convergence
    // =========================================================================
    static inline __m128 screlu_sse(__m128 x) {
        __m128 zero = _mm_setzero_ps();
        __m128 one = _mm_set1_ps(1.0f);
        __m128 clamped = _mm_max_ps(zero, _mm_min_ps(x, one));
        return _mm_mul_ps(clamped, clamped);  // square it
    }

    // =========================================================================
    // Feature index helpers
    // =========================================================================
    int featureIndex(PieceType pt, Color pc, int rank, int col) {
        int pieceOffset = 0;
        switch (pt) {
            case PieceType::Pawn:   pieceOffset = 0; break;
            case PieceType::Knight: pieceOffset = 1; break;
            case PieceType::Bishop: pieceOffset = 2; break;
            case PieceType::Rook:   pieceOffset = 3; break;
            case PieceType::Queen:  pieceOffset = 4; break;
            case PieceType::King:   pieceOffset = 5; break;
            default: return -1;
        }
        int pieceIndex = (pc == Color::White) ? pieceOffset : pieceOffset + 6;
        return pieceIndex * 64 + rank * 8 + col;
    }

    int mirrorFeature(int feature) {
        int pieceIndex = feature / 64;
        int squareIndex = feature % 64;
        int rank = squareIndex / 8;
        int col = squareIndex % 8;

        int mirroredPieceIndex = (pieceIndex < 6) ? (pieceIndex + 6) : (pieceIndex - 6);
        int mirroredRank = 7 - rank;

        return mirroredPieceIndex * 64 + mirroredRank * 8 + col;
    }

    Network::Network() {
        randomizeWeights();
    }

    int Network::evaluate(const Board& board) {
        Accumulator acc;
        refreshAccumulator(board, acc);
        return forward(acc, board.turn);
    }

    // =========================================================================
    // refreshAccumulator: SIMD-accelerated accumulator computation
    // =========================================================================
    void Network::refreshAccumulator(const Board& board, Accumulator& acc) {
        for (int j = 0; j < L1_SIZE; j += 4) {
            __m128 bias = _mm_load_ps(&L1_biases[j]);
            _mm_store_ps(&acc.white[j], bias);
            _mm_store_ps(&acc.black[j], bias);
        }

        for (int rank = 0; rank < 8; ++rank) {
            for (int col = 0; col < 8; ++col) {
                Piece piece = board.squares[rank][col];
                if (piece.isNone() || piece.isDuck()) continue;

                int wFeature = featureIndex(piece.type, piece.color, rank, col);
                int bFeature = mirrorFeature(wFeature);

                const float* wWeights = L1_weights[wFeature].data();
                float* wAcc = acc.white.data();
                for (int j = 0; j < L1_SIZE; j += 4) {
                    __m128 a = _mm_load_ps(&wAcc[j]);
                    __m128 w = _mm_load_ps(&wWeights[j]);
                    _mm_store_ps(&wAcc[j], _mm_add_ps(a, w));
                }

                const float* bWeights = L1_weights[bFeature].data();
                float* bAcc = acc.black.data();
                for (int j = 0; j < L1_SIZE; j += 4) {
                    __m128 a = _mm_load_ps(&bAcc[j]);
                    __m128 w = _mm_load_ps(&bWeights[j]);
                    _mm_store_ps(&bAcc[j], _mm_add_ps(a, w));
                }
            }
        }

        acc.valid = true;
    }

    // =========================================================================
    // Incremental feature add/remove — SIMD accelerated
    // =========================================================================
    void Network::addFeature(int feature, Accumulator& acc) {
        int mirrored = mirrorFeature(feature);

        const float* wWeights = L1_weights[feature].data();
        float* wAcc = acc.white.data();
        for (int j = 0; j < L1_SIZE; j += 4) {
            __m128 a = _mm_load_ps(&wAcc[j]);
            __m128 w = _mm_load_ps(&wWeights[j]);
            _mm_store_ps(&wAcc[j], _mm_add_ps(a, w));
        }

        const float* bWeights = L1_weights[mirrored].data();
        float* bAcc = acc.black.data();
        for (int j = 0; j < L1_SIZE; j += 4) {
            __m128 a = _mm_load_ps(&bAcc[j]);
            __m128 w = _mm_load_ps(&bWeights[j]);
            _mm_store_ps(&bAcc[j], _mm_add_ps(a, w));
        }
    }

    void Network::removeFeature(int feature, Accumulator& acc) {
        int mirrored = mirrorFeature(feature);

        const float* wWeights = L1_weights[feature].data();
        float* wAcc = acc.white.data();
        for (int j = 0; j < L1_SIZE; j += 4) {
            __m128 a = _mm_load_ps(&wAcc[j]);
            __m128 w = _mm_load_ps(&wWeights[j]);
            _mm_store_ps(&wAcc[j], _mm_sub_ps(a, w));
        }

        const float* bWeights = L1_weights[mirrored].data();
        float* bAcc = acc.black.data();
        for (int j = 0; j < L1_SIZE; j += 4) {
            __m128 a = _mm_load_ps(&bAcc[j]);
            __m128 w = _mm_load_ps(&bWeights[j]);
            _mm_store_ps(&bAcc[j], _mm_sub_ps(a, w));
        }
    }

    // =========================================================================
    // forward: SIMD-optimized forward pass with SCReLU activation
    // CHANGED: All activations now use SCReLU instead of ClippedReLU
    // =========================================================================
    int Network::forward(const Accumulator& acc, Color sideToMove) {
        // Build 512-element input with SCReLU — SIMD accelerated
        alignas(16) float input[L1_SIZE * 2];

        const auto& stmAcc = (sideToMove == Color::White) ? acc.white : acc.black;
        const auto& oppAcc = (sideToMove == Color::White) ? acc.black : acc.white;

        // SCReLU on STM accumulator
        for (int i = 0; i < L1_SIZE; i += 4) {
            __m128 val = _mm_load_ps(&stmAcc[i]);
            _mm_store_ps(&input[i], screlu_sse(val));
        }
        // SCReLU on opponent accumulator
        for (int i = 0; i < L1_SIZE; i += 4) {
            __m128 val = _mm_load_ps(&oppAcc[i]);
            _mm_store_ps(&input[L1_SIZE + i], screlu_sse(val));
        }

        // L2: 512 -> 64 (SIMD dot products)
        alignas(16) float l2Out[L2_SIZE];
        for (int j = 0; j < L2_SIZE; ++j) {
            __m128 sum = _mm_setzero_ps();
            for (int i = 0; i < L1_SIZE * 2; i += 4) {
                __m128 inp = _mm_load_ps(&input[i]);
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
            // CHANGED: SCReLU activation
            float clamped = std::max(0.0f, std::min(val, 1.0f));
            l2Out[j] = clamped * clamped;
        }

        // L3: 64 -> 64 (SIMD dot products)
        alignas(16) float l3Out[L3_SIZE];
        for (int j = 0; j < L3_SIZE; ++j) {
            __m128 sum = _mm_setzero_ps();
            for (int i = 0; i < L2_SIZE; i += 4) {
                __m128 inp = _mm_load_ps(&l2Out[i]);
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
            // CHANGED: SCReLU activation
            float clamped = std::max(0.0f, std::min(val, 1.0f));
            l3Out[j] = clamped * clamped;
        }

        // Output: 64 -> 1 (SIMD dot product)
        __m128 outputSum = _mm_setzero_ps();
        for (int i = 0; i < L3_SIZE; i += 4) {
            __m128 inp = _mm_load_ps(&l3Out[i]);
            __m128 w = _mm_load_ps(&output_weights[i]);
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

        // Validate file size matches expected weight data size
        file.seekg(0, std::ios::end);
        auto fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        const std::streamsize expectedSize =
            sizeof(L1_weights) + sizeof(L1_biases) +
            sizeof(L2_weights) + sizeof(L2_biases) +
            sizeof(L3_weights) + sizeof(L3_biases) +
            sizeof(output_weights) + sizeof(output_bias);

        if (fileSize < static_cast<std::streampos>(expectedSize)) {
            return false;
        }

        file.read(reinterpret_cast<char*>(L1_weights.data()), sizeof(L1_weights));
        if (!file.good()) return false;
        file.read(reinterpret_cast<char*>(L1_biases.data()), sizeof(L1_biases));
        if (!file.good()) return false;
        file.read(reinterpret_cast<char*>(L2_weights.data()), sizeof(L2_weights));
        if (!file.good()) return false;
        file.read(reinterpret_cast<char*>(L2_biases.data()), sizeof(L2_biases));
        if (!file.good()) return false;
        file.read(reinterpret_cast<char*>(L3_weights.data()), sizeof(L3_weights));
        if (!file.good()) return false;
        file.read(reinterpret_cast<char*>(L3_biases.data()), sizeof(L3_biases));
        if (!file.good()) return false;
        file.read(reinterpret_cast<char*>(output_weights.data()), sizeof(output_weights));
        if (!file.good()) return false;
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

} // namespace NNUE
