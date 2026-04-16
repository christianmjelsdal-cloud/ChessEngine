#include "DuckNNUE.h"
#include <fstream>
#include <random>
#include <cmath>
#include <algorithm>
#include <cstring>

#ifdef _MSC_VER
#  include <intrin.h>
#  include <immintrin.h>
#else
#include <immintrin.h>
#endif

namespace DuckNNUE {

    // SSE helper: SCReLU on 4 floats
    static inline __m128 screlu_sse(__m128 x) {
        __m128 zero = _mm_setzero_ps();
        __m128 one = _mm_set1_ps(1.0f);
        __m128 clamped = _mm_max_ps(zero, _mm_min_ps(x, one));
        return _mm_mul_ps(clamped, clamped);
    }

    // AVX2 helper: SCReLU on 8 floats
    static inline __m256 screlu_avx(__m256 x) {
        __m256 zero = _mm256_setzero_ps();
        __m256 one  = _mm256_set1_ps(1.0f);
        __m256 c = _mm256_min_ps(_mm256_max_ps(x, zero), one);
        return _mm256_mul_ps(c, c);
    }

    Network::Network()
        : L1_weights_q(std::make_unique<std::array<std::array<int16_t, L1_SIZE>, NUM_FEATURES>>())
    {
        randomizeWeights();
        transposeWeights();
        quantizeWeights();
    }

    void Network::transposeWeights() {
        for (int j = 0; j < L2_SIZE; ++j)
            for (int i = 0; i < L1_SIZE * 2; ++i)
                L2_weights_T[j][i] = L2_weights[i][j];
        for (int j = 0; j < L3_SIZE; ++j)
            for (int i = 0; i < L2_SIZE; ++i)
                L3_weights_T[j][i] = L3_weights[i][j];
    }

    int Network::evaluate(const Board& board) const {
        Accumulator acc;
        refreshAccumulator(board, acc);
        return forward(acc, board.turn);
    }

    void Network::refreshAccumulator(const Board& board, Accumulator& acc) const {
        // Initialize with biases
        for (int j = 0; j < L1_SIZE; j += 8) {
            __m256 bias = _mm256_loadu_ps(&L1_biases[j]);
            _mm256_storeu_ps(&acc.white[j], bias);
            _mm256_storeu_ps(&acc.black[j], bias);
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
                for (int j = 0; j < L1_SIZE; j += 8) {
                    _mm256_storeu_ps(&wAcc[j], _mm256_add_ps(_mm256_loadu_ps(&wAcc[j]), _mm256_loadu_ps(&wWeights[j])));
                }

                const float* bWeights = L1_weights[bFeature].data();
                float* bAcc = acc.black.data();
                for (int j = 0; j < L1_SIZE; j += 8) {
                    _mm256_storeu_ps(&bAcc[j], _mm256_add_ps(_mm256_loadu_ps(&bAcc[j]), _mm256_loadu_ps(&bWeights[j])));
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

    void Network::addFeature(int feature, Accumulator& acc) const {
        int mirrored = mirrorDuckFeature(feature);  // handles both standard and duck features

        const float* wWeights = L1_weights[feature].data();
        float* wAcc = acc.white.data();
        for (int j = 0; j < L1_SIZE; j += 8) {
                    _mm256_storeu_ps(&wAcc[j], _mm256_add_ps(_mm256_loadu_ps(&wAcc[j]), _mm256_loadu_ps(&wWeights[j])));
                }

        const float* bWeights = L1_weights[mirrored].data();
        float* bAcc = acc.black.data();
        for (int j = 0; j < L1_SIZE; j += 8) {
                    _mm256_storeu_ps(&bAcc[j], _mm256_add_ps(_mm256_loadu_ps(&bAcc[j]), _mm256_loadu_ps(&bWeights[j])));
                }
    }

    void Network::removeFeature(int feature, Accumulator& acc) const {
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

    int Network::forward(const Accumulator& acc, Color sideToMove) const {
        // Build input with SCReLU
        alignas(16) float input[L1_SIZE * 2];
        const auto& stmAcc = (sideToMove == Color::White) ? acc.white : acc.black;
        const auto& oppAcc = (sideToMove == Color::White) ? acc.black : acc.white;
        for (int i = 0; i < L1_SIZE; i += 8) {
            _mm256_storeu_ps(&input[i],           screlu_avx(_mm256_loadu_ps(&stmAcc[i])));
            _mm256_storeu_ps(&input[L1_SIZE + i], screlu_avx(_mm256_loadu_ps(&oppAcc[i])));
        }

        // L2 — use transposed weights for cache-friendly access
        alignas(16) float l2Out[L2_SIZE];
        for (int j = 0; j < L2_SIZE; ++j) {
            const float* w = L2_weights_T[j].data();
            __m128 sum = _mm_setzero_ps();
            for (int i = 0; i < L1_SIZE * 2; i += 8) {
                __m256 s = _mm256_mul_ps(_mm256_loadu_ps(&input[i]), _mm256_loadu_ps(&w[i]));
                sum = _mm_add_ps(sum, _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1)));
            }
            // horizontal sum
            __m128 shuf = _mm_movehdup_ps(sum);
            sum = _mm_add_ps(sum, shuf);
            shuf = _mm_movehl_ps(shuf, sum);
            sum = _mm_add_ss(sum, shuf);
            float val = _mm_cvtss_f32(sum) + L2_biases[j];
            float c = val < 0.f ? 0.f : val > 1.f ? 1.f : val;
            l2Out[j] = c * c;
        }

        // L3 — use transposed weights
        alignas(16) float l3Out[L3_SIZE];
        for (int j = 0; j < L3_SIZE; ++j) {
            const float* w = L3_weights_T[j].data();
            __m128 sum = _mm_setzero_ps();
            for (int i = 0; i < L2_SIZE; i += 8) {
                __m256 s = _mm256_mul_ps(_mm256_loadu_ps(&l2Out[i]), _mm256_loadu_ps(&w[i]));
                sum = _mm_add_ps(sum, _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1)));
            }
            __m128 shuf = _mm_movehdup_ps(sum);
            sum = _mm_add_ps(sum, shuf);
            shuf = _mm_movehl_ps(shuf, sum);
            sum = _mm_add_ss(sum, shuf);
            float val = _mm_cvtss_f32(sum) + L3_biases[j];
            float c = val < 0.f ? 0.f : val > 1.f ? 1.f : val;
            l3Out[j] = c * c;
        }

        // Output
        __m128 sum = _mm_setzero_ps();
        for (int i = 0; i < L3_SIZE; i += 4)
            sum = _mm_add_ps(sum, _mm_mul_ps(_mm_loadu_ps(&l3Out[i]), _mm_loadu_ps(&output_weights[i])));
        __m128 shuf = _mm_movehdup_ps(sum);
        sum = _mm_add_ps(sum, shuf);
        shuf = _mm_movehl_ps(shuf, sum);
        sum = _mm_add_ss(sum, shuf);
        return static_cast<int>((_mm_cvtss_f32(sum) + output_bias) * 400.f);
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

        if (file.good()) { transposeWeights(); quantizeWeights(); }
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
        transposeWeights();
        quantizeWeights();
    }

    // =========================================================================
    // INT16 quantization
    // =========================================================================
    void Network::quantizeWeights() {
        if (!L1_weights_q)
            L1_weights_q = std::make_unique<std::array<std::array<int16_t, L1_SIZE>, NUM_FEATURES>>();

        // L1: float → int16 (scale QA=256)
        for (int f = 0; f < NUM_FEATURES; ++f)
            for (int j = 0; j < L1_SIZE; ++j) {
                int q = static_cast<int>(std::round(L1_weights[f][j] * QA));
                q = std::max(-32768, std::min(32767, q));
                (*L1_weights_q)[f][j] = static_cast<int16_t>(q);
            }
        for (int j = 0; j < L1_SIZE; ++j) {
            int q = static_cast<int>(std::round(L1_biases[j] * QA));
            q = std::max(-32768, std::min(32767, q));
            L1_biases_q[j] = static_cast<int16_t>(q);
        }

        // L2: float → int8 transposed (scale QW_L2)
        for (int j = 0; j < L2_SIZE; ++j) {
            for (int i = 0; i < L1_SIZE * 2; ++i) {
                int q = static_cast<int>(std::round(L2_weights_T[j][i] * QW_L2));
                q = std::max(-127, std::min(127, q));
                L2_weights_T_q[j][i] = static_cast<int8_t>(q);
            }
            L2_biases_q[j] = static_cast<int32_t>(std::round(L2_biases[j] * QA_ACT * QW_L2));
        }

        // L3: float → int8 transposed (scale QW_L3)
        for (int j = 0; j < L3_SIZE; ++j) {
            for (int i = 0; i < L2_SIZE; ++i) {
                int q = static_cast<int>(std::round(L3_weights_T[j][i] * QW_L3));
                q = std::max(-127, std::min(127, q));
                L3_weights_T_q[j][i] = static_cast<int8_t>(q);
            }
            L3_biases_q[j] = static_cast<int32_t>(std::round(L3_biases[j] * QA_ACT * QW_L3));
        }
    }

    void Network::addFeatureQ(int feature, QAccumulator& acc) const {
        if (!L1_weights_q || feature < 0 || feature >= NUM_FEATURES) return;
        int mir = mirrorDuckFeature(feature);
        if (mir < 0 || mir >= NUM_FEATURES) return;
        const int16_t* wW = (*L1_weights_q)[feature].data();
        const int16_t* bW = (*L1_weights_q)[mir].data();
        int16_t* wA = acc.white.data(); int16_t* bA = acc.black.data();
        for (int j = 0; j < L1_SIZE; j += 16) {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&wA[j]),
                _mm256_adds_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(&wA[j])),
                                  _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&wW[j]))));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&bA[j]),
                _mm256_adds_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(&bA[j])),
                                  _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&bW[j]))));
        }
    }

    void Network::removeFeatureQ(int feature, QAccumulator& acc) const {
        if (!L1_weights_q || feature < 0 || feature >= NUM_FEATURES) return;
        int mir = mirrorDuckFeature(feature);
        if (mir < 0 || mir >= NUM_FEATURES) return;
        const int16_t* wW = (*L1_weights_q)[feature].data();
        const int16_t* bW = (*L1_weights_q)[mir].data();
        int16_t* wA = acc.white.data(); int16_t* bA = acc.black.data();
        for (int j = 0; j < L1_SIZE; j += 16) {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&wA[j]),
                _mm256_subs_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(&wA[j])),
                                  _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&wW[j]))));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&bA[j]),
                _mm256_subs_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(&bA[j])),
                                  _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&bW[j]))));
        }
    }

    int Network::forwardQ(const QAccumulator& acc, Color sideToMove) const {
        const auto& stm = (sideToMove == Color::White) ? acc.white : acc.black;
        const auto& opp = (sideToMove == Color::White) ? acc.black : acc.white;

        // Dequantize INT16 accumulator → float SCReLU
        // Use thread_local to avoid stack alignment issues in worker threads
        static thread_local float input[L1_SIZE * 2];
        static thread_local float l2Out[L2_SIZE];
        static thread_local float l3Out[L3_SIZE];

        // AVX2 SCReLU dequantize: int16 -> float, clamp [0,QA], square, scale by 1/QA^2
        const float invQA2 = 1.0f / (static_cast<float>(QA) * static_cast<float>(QA));
        const __m256 vZero   = _mm256_setzero_ps();
        const __m256 vQA     = _mm256_set1_ps(static_cast<float>(QA));
        const __m256 vInvQA2 = _mm256_set1_ps(invQA2);
        for (int i = 0; i < L1_SIZE; i += 8) {
            // Load 8 int16 -> int32 -> float for stm
            __m128i si16s = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&stm[i]));
            __m256  sf    = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(si16s));
            sf = _mm256_min_ps(_mm256_max_ps(sf, vZero), vQA);
            sf = _mm256_mul_ps(_mm256_mul_ps(sf, sf), vInvQA2);
            _mm256_storeu_ps(&input[i], sf);

            // Load 8 int16 -> int32 -> float for opp
            __m128i oi16s = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&opp[i]));
            __m256  of    = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(oi16s));
            of = _mm256_min_ps(_mm256_max_ps(of, vZero), vQA);
            of = _mm256_mul_ps(_mm256_mul_ps(of, of), vInvQA2);
            _mm256_storeu_ps(&input[L1_SIZE + i], of);
        }

        // L2 — INT8 quantized weights: 32 values per AVX2 op (4x vs float)
        // input_q: uint8 SCReLU output [0,127], L2_weights_T_q: int8 [-127,127]
        // Use _mm256_maddubs_epi16: (uint8 * int8) -> int16, 32 per op
        // Dequantize SCReLU output to uint8 [0, QA_ACT=127] for INT8 path
        static thread_local uint8_t input_q[L1_SIZE * 2];
        {
            const __m256i vZero8 = _mm256_setzero_si256();
            const __m256i vQA8   = _mm256_set1_epi16(QA_ACT);
            for (int i = 0; i < L1_SIZE; i += 8) {
                __m128i si16s = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&stm[i]));
                __m256i si32  = _mm256_cvtepi16_epi32(si16s);
                // SCReLU: clamp [0,QA], square, scale to [0,QA_ACT]
                // Approximate: clamp to [0,QA], then (v * QA_ACT) / QA
                __m256i clamped = _mm256_max_epi32(vZero8, _mm256_min_epi32(si32, _mm256_set1_epi32(QA)));
                // (clamped * clamped * QA_ACT) / (QA * QA) — use float for this step
                __m256 sf = _mm256_cvtepi32_ps(clamped);
                sf = _mm256_mul_ps(_mm256_mul_ps(sf, sf), _mm256_set1_ps((float)QA_ACT / ((float)QA * QA)));
                __m128i r = _mm256_cvtps_epi32(sf);  // truncate to int32
                // Pack to uint8 via int16
                __m128i r16 = _mm_packs_epi32(r, r);
                __m128i r8  = _mm_packus_epi16(r16, r16);
                _mm_storel_epi64(reinterpret_cast<__m128i*>(&input_q[i]), r8);

                __m128i oi16s = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&opp[i]));
                __m256i oi32  = _mm256_cvtepi16_epi32(oi16s);
                __m256i oclamped = _mm256_max_epi32(vZero8, _mm256_min_epi32(oi32, _mm256_set1_epi32(QA)));
                __m256 of = _mm256_cvtepi32_ps(oclamped);
                of = _mm256_mul_ps(_mm256_mul_ps(of, of), _mm256_set1_ps((float)QA_ACT / ((float)QA * QA)));
                __m128i or32 = _mm256_cvtps_epi32(of);
                __m128i or16 = _mm_packs_epi32(or32, or32);
                __m128i or8  = _mm_packus_epi16(or16, or16);
                _mm_storel_epi64(reinterpret_cast<__m128i*>(&input_q[L1_SIZE + i]), or8);
            }
        }

        for (int j = 0; j < L2_SIZE; ++j) {
            const int8_t* w = L2_weights_T_q[j].data();
            __m256i sum32 = _mm256_setzero_si256();
            for (int i = 0; i < L1_SIZE * 2; i += 32) {
                __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&input_q[i]));
                __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&w[i]));
                // maddubs: (uint8 * int8) -> int16, then madd: int16 pairs -> int32
                __m256i prod = _mm256_maddubs_epi16(a, b);
                sum32 = _mm256_add_epi32(sum32, _mm256_madd_epi16(prod, _mm256_set1_epi16(1)));
            }
            // Horizontal sum of 8 int32 lanes
            __m128i lo = _mm256_castsi256_si128(sum32);
            __m128i hi = _mm256_extracti128_si256(sum32, 1);
            __m128i s4 = _mm_add_epi32(lo, hi);
            s4 = _mm_hadd_epi32(s4, s4);
            s4 = _mm_hadd_epi32(s4, s4);
            int32_t raw = _mm_cvtsi128_si32(s4) + L2_biases_q[j];
            // Dequantize: raw / (QA_ACT * QW_L2), then SCReLU
            float val = static_cast<float>(raw) / (static_cast<float>(QA_ACT) * QW_L2);
            float c = val < 0.f ? 0.f : val > 1.f ? 1.f : val;
            l2Out[j] = c * c;
        }

        // L3 — INT8 quantized weights (L2_SIZE=128 inputs, L3_SIZE=64 outputs)
        // Quantize l2Out (float [0,1]) to uint8 [0,127] for maddubs
        static thread_local uint8_t l2Out_q[L2_SIZE];
        {
            const __m256 vScale = _mm256_set1_ps(static_cast<float>(QA_ACT));
            for (int i = 0; i < L2_SIZE; i += 8) {
                __m256 v = _mm256_mul_ps(_mm256_loadu_ps(&l2Out[i]), vScale);
                __m128i i32 = _mm256_cvtps_epi32(v);
                __m128i i16 = _mm_packs_epi32(i32, i32);
                __m128i u8  = _mm_packus_epi16(i16, i16);
                _mm_storel_epi64(reinterpret_cast<__m128i*>(&l2Out_q[i]), u8);
            }
        }
        for (int j = 0; j < L3_SIZE; ++j) {
            const int8_t* w = L3_weights_T_q[j].data();
            // L2_SIZE=128: two 64-element AVX2 passes
            __m256i sum32 = _mm256_setzero_si256();
            for (int i = 0; i < L2_SIZE; i += 32) {
                __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&l2Out_q[i]));
                __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&w[i]));
                __m256i prod = _mm256_maddubs_epi16(a, b);
                sum32 = _mm256_add_epi32(sum32, _mm256_madd_epi16(prod, _mm256_set1_epi16(1)));
            }
            __m128i lo = _mm256_castsi256_si128(sum32);
            __m128i hi = _mm256_extracti128_si256(sum32, 1);
            __m128i s4 = _mm_add_epi32(lo, hi);
            s4 = _mm_hadd_epi32(s4, s4);
            s4 = _mm_hadd_epi32(s4, s4);
            int32_t raw = _mm_cvtsi128_si32(s4) + L3_biases_q[j];
            float val = static_cast<float>(raw) / (static_cast<float>(QA_ACT) * QW_L3);
            float c = val < 0.f ? 0.f : val > 1.f ? 1.f : val;
            l3Out[j] = c * c;
        }

        // Output
        __m256 sum = _mm256_setzero_ps();
        for (int i = 0; i < L3_SIZE; i += 8)
            sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_loadu_ps(&l3Out[i]), _mm256_loadu_ps(&output_weights[i])));
        __m128 s = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
        __m128 shuf = _mm_movehdup_ps(s); s = _mm_add_ps(s, shuf);
        s = _mm_add_ss(s, _mm_movehl_ps(shuf, s));
        return static_cast<int>((_mm_cvtss_f32(s) + output_bias) * 400.f);
    }

    int Network::evaluateQ(const Board& board) const {
        // INT16 quantized path — refreshAccumulatorQ (scalar) + forwardQ (AVX2 float)
        static thread_local QAccumulator acc;
        refreshAccumulatorQ(board, acc);
        return forwardQ(acc, board.turn);
    }

    void Network::refreshAccumulatorQ(const Board& board, QAccumulator& acc) const {
        if (!L1_weights_q) { acc.valid = false; return; }

        // Init with quantized biases (AVX2: 16 int16 per op)
        const int16_t* bias = L1_biases_q.data();
        int16_t* wA = acc.white.data();
        int16_t* bA = acc.black.data();
        for (int j = 0; j < L1_SIZE; j += 16) {
            __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&bias[j]));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&wA[j]), b);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&bA[j]), b);
        }

        // Piece features (AVX2)
        for (int rank = 0; rank < 8; ++rank) {
            for (int col = 0; col < 8; ++col) {
                Piece piece = board.squares[rank][col];
                if (piece.isNone() || piece.isDuck()) continue;
                int wFeat = NNUE::featureIndex(piece.type, piece.color, rank, col);
                if (wFeat < 0 || wFeat >= NUM_FEATURES) continue;
                int bFeat = NNUE::mirrorFeature(wFeat);
                if (bFeat < 0 || bFeat >= NUM_FEATURES) continue;
                const int16_t* wW = (*L1_weights_q)[wFeat].data();
                const int16_t* bW = (*L1_weights_q)[bFeat].data();
                for (int j = 0; j < L1_SIZE; j += 16) {
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(&wA[j]),
                        _mm256_adds_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(&wA[j])),
                                          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&wW[j]))));
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(&bA[j]),
                        _mm256_adds_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(&bA[j])),
                                          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&bW[j]))));
                }
            }
        }

        // Duck feature (AVX2)
        if (board.isDuckChess && board.duckSquare.isValid()) {
            int dFeat  = duckFeatureIndex(board.duckSquare.rank, board.duckSquare.col);
            int dFeatM = mirrorDuckFeature(dFeat);
            const int16_t* dW  = (*L1_weights_q)[dFeat].data();
            const int16_t* dWM = (*L1_weights_q)[dFeatM].data();
            for (int j = 0; j < L1_SIZE; j += 16) {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&wA[j]),
                    _mm256_adds_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(&wA[j])),
                                      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&dW[j]))));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&bA[j]),
                    _mm256_adds_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(&bA[j])),
                                      _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&dWM[j]))));
            }
        }
        acc.valid = true;
    }

} // namespace DuckNNUE
