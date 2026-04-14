#include "NNUE.h"
#include "Bitboard.h"
#include <fstream>
#include <random>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <cassert>

// SIMD vectorization
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <immintrin.h>
#endif

namespace NNUE {

    using BB::popLsb;

    // 2.11: AVX2 loops require L1_SIZE to be multiple of 16 (int16) and 8 (float)
    static_assert(L1_SIZE % 16 == 0, "L1_SIZE must be a multiple of 16 for AVX2 int16/float alignment");


    // =========================================================================
    // SSE helper: SCReLU (Squared Clipped ReLU) on 4 floats: max(0, min(x, 1))^2
    // SCReLU preserves more gradient information and improves training convergence
    // =========================================================================
    static inline __m128 screlu_sse(__m128 x) {
        __m128 zero = _mm_setzero_ps();
        __m128 one = _mm_set1_ps(1.0f);
        __m128 clamped = _mm_max_ps(zero, _mm_min_ps(x, one));
        return _mm_mul_ps(clamped, clamped);  // square it
    }

    // AVX2 helper: SCReLU on 8 floats: max(0, min(x, 1))^2
    static inline __m256 screlu_avx(__m256 x) {
        __m256 zero = _mm256_setzero_ps();
        __m256 one  = _mm256_set1_ps(1.0f);
        __m256 clamped = _mm256_min_ps(_mm256_max_ps(x, zero), one);
        return _mm256_mul_ps(clamped, clamped);
    }

    // AVX2 horizontal sum: reduce 8 floats to 1
    static inline float hsum_avx(__m256 v) {
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        lo = _mm_add_ps(lo, hi);
        __m128 shuf = _mm_movehdup_ps(lo);
        __m128 sums = _mm_add_ps(lo, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        return _mm_cvtss_f32(sums);
    }

    // AVX2 horizontal sum: reduce 8 int32 to 1
    static inline int32_t hsum_epi32(__m256i v) {
        __m128i lo = _mm256_castsi256_si128(v);
        __m128i hi = _mm256_extracti128_si256(v, 1);
        lo = _mm_add_epi32(lo, hi);            // 4 int32
        __m128i shuf = _mm_shuffle_epi32(lo, _MM_SHUFFLE(1,0,3,2));
        lo = _mm_add_epi32(lo, shuf);          // 2 int32
        shuf = _mm_shuffle_epi32(lo, _MM_SHUFFLE(0,1,0,1));
        lo = _mm_add_epi32(lo, shuf);          // 1 int32
        return _mm_cvtsi128_si32(lo);
    }

    // Fast scalar exp approximation using range reduction + polynomial
    // Max relative error ~1.7e-4 in [-88, 88] — plenty accurate for WDL softmax
    static inline float fast_expf(float x) {
        if (x < -87.0f) return 0.0f;
        if (x > 88.0f) return 3.4028235e+38f;

        const float log2e = 1.44269504089f;
        float t = x * log2e;
        float n = std::floor(t);
        float f = t - n;

        // Minimax polynomial for 2^f, f in [0, 1)
        float p = 1.3534167e-2f;
        p = p * f + 5.2011464e-2f;
        p = p * f + 2.4114610e-1f;
        p = p * f + 6.9315308e-1f;
        p = p * f + 1.0f;

        // 2^n via IEEE 754 bit manipulation
        int32_t ni = static_cast<int32_t>(n);
        union { float fv; int32_t iv; } u;
        u.iv = (ni + 127) << 23;

        return p * u.fv;
    }

    // =========================================================================
    // HalfKAv2 feature index computation
    // =========================================================================
    int halfKAv2Feature(int king_sq, PieceType pt, Color pieceColor, int piece_sq, Color perspective) {
        if (pt == PieceType::King || pt == PieceType::None) return -1;

        // Piece type index: 0-4 for own pieces, 5-9 for opponent
        int ptIdx = static_cast<int>(pt) - 1;  // Pawn=0, Knight=1, Bishop=2, Rook=3, Queen=4
        bool isOwn = (pieceColor == perspective);
        if (!isOwn) ptIdx += 5;

        // Mirror squares for black perspective
        int ksq = king_sq;
        int psq = piece_sq;
        if (perspective == Color::Black) {
            ksq = ksq ^ 56;  // flip rank
            psq = psq ^ 56;
        }

        return ksq * 640 + ptIdx * 64 + psq;
    }

    // =========================================================================
    // Legacy 768-encoding feature index (for data file compatibility)
    // =========================================================================
    int featureIndex768(PieceType pt, Color pc, int rank, int col) {
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

    Network::Network()
        : L1_weights(std::make_unique<std::array<std::array<float, L1_SIZE>, NUM_FEATURES>>()),
          L1_weights_q(std::make_unique<std::array<std::array<int16_t, L1_SIZE>, NUM_FEATURES>>())
    {
        randomizeWeights();
        // Ensure quantized weights are always valid — evaluateQ() requires L1_weights_q
        // to be initialized even when using random weights (no file loaded).
        transposeWeights();
        quantizeWeights();
    }

    // =========================================================================
    // computePhase: material-based game phase (0.0=endgame, 1.0=opening)
    // =========================================================================
    float Network::computePhase(const Board& board) {
        // AUDIT FIX E-5: Use incrementally maintained phase from Board
        // instead of scanning all 64 squares every evaluation
        static const int TOTAL_PHASE = 24;
        int p = std::max(0, std::min(board.phase, TOTAL_PHASE));
        return p / static_cast<float>(TOTAL_PHASE);
    }

    int Network::evaluate(const Board& board) const {
        Accumulator acc;
        refreshAccumulator(board, acc);
        float phase = computePhase(board);
        return forward(acc, board.turn, phase);
    }

    // =========================================================================
    // refreshAccumulator: SIMD-accelerated HalfKAv2 accumulator computation
    // =========================================================================
    void Network::refreshAccumulator(const Board& board, Accumulator& acc) const {
        // Use Board's cached king squares (avoids 64-square mailbox scan)
        int wKingSq = squareIndex(board.whiteKingSq.rank, board.whiteKingSq.col);
        int bKingSq = squareIndex(board.blackKingSq.rank, board.blackKingSq.col);
        acc.whiteKingSq = wKingSq;
        acc.blackKingSq = bKingSq;

        // IMPORTANT: Accumulator arrays must be alignas(32) for AVX2 _mm256_loadu_ps/_mm256_storeu_ps
        // (Currently alignas(64) in NNUE.h — satisfies this requirement)
        // Initialize accumulators with biases (AVX2: 8 floats per op)
        for (int j = 0; j < L1_SIZE; j += 8) {
            __m256 bias = _mm256_loadu_ps(&L1_biases[j]);
            _mm256_storeu_ps(&acc.white[j], bias);
            _mm256_storeu_ps(&acc.black[j], bias);
        }

        // FIX 2.20: Iterate occupied squares via popLsb (O(pieces) vs O(64) mailbox scan)
        {
            Bitboard occ = board.occupied();
            while (occ) {
                int sq = popLsb(occ);
                int rank = sq / 8;
                int col = sq % 8;
                Piece piece = board.squares[rank][col];
                if (piece.isNone() || piece.isDuck() || piece.type == PieceType::King)
                    continue;

                // White perspective feature
                int wFeature = halfKAv2Feature(wKingSq, piece.type, piece.color, sq, Color::White);
                if (wFeature >= 0) {
                    const float* wWeights = (*L1_weights)[wFeature].data();
                    float* wAcc = acc.white.data();
                    for (int j = 0; j < L1_SIZE; j += 8) {
                        __m256 a = _mm256_loadu_ps(&wAcc[j]);
                        __m256 w = _mm256_loadu_ps(&wWeights[j]);
                        _mm256_storeu_ps(&wAcc[j], _mm256_add_ps(a, w));
                    }
                }

                // Black perspective feature
                int bFeature = halfKAv2Feature(bKingSq, piece.type, piece.color, sq, Color::Black);
                if (bFeature >= 0) {
                    const float* bWeights = (*L1_weights)[bFeature].data();
                    float* bAcc = acc.black.data();
                    for (int j = 0; j < L1_SIZE; j += 8) {
                        __m256 a = _mm256_loadu_ps(&bAcc[j]);
                        __m256 w = _mm256_loadu_ps(&bWeights[j]);
                        _mm256_storeu_ps(&bAcc[j], _mm256_add_ps(a, w));
                    }
                }
            }
        }
        acc.valid = true;
    }

    // =========================================================================
    // incrementalUpdate: efficient move-by-move accumulator update
    // NOTE (M2/M3): This function does NOT handle en passant (captured pawn
    //   is on a different square than toSq) or promotions (adds Pawn at toSq
    //   instead of promoted piece). Currently safe because Engine.cpp uses
    //   refreshAccumulatorQ for all special moves (isSpecial flag). If the
    //   caller logic changes, these cases must be handled here.
    // =========================================================================
    void Network::incrementalUpdate(const Board& board, Accumulator& acc,
                                    int fromRank, int fromCol, int toRank, int toCol,
                                    PieceType movedPiece, Color movedColor,
                                    PieceType capturedPiece, Color capturedColor) const {
        // FIX H-1: Runtime guard — en passant and promotions cannot be handled
        // incrementally. Fall back to full refresh instead of assert-only protection.
        {
            bool isPromotion = (movedPiece == PieceType::Pawn && (toRank == 0 || toRank == 7));
            bool isEnPassant = (movedPiece == PieceType::Pawn && fromCol != toCol
                                && capturedPiece == PieceType::None);
            if (isPromotion || isEnPassant) {
                refreshAccumulator(board, acc);
                return;
            }
        }
        // FIX C-2: Auto-refresh accumulator on king moves (all HalfKAv2 features change)
        if (movedPiece == PieceType::King) {
            refreshAccumulator(board, acc);
            return;
        }

        int fromSq = squareIndex(fromRank, fromCol);
        int toSq = squareIndex(toRank, toCol);

        // For each perspective (white, black)
        for (int persp = 0; persp < 2; ++persp) {
            Color perspective = static_cast<Color>(persp);
            int kingSq = (perspective == Color::White) ? acc.whiteKingSq : acc.blackKingSq;
            float* accumulator = (perspective == Color::White) ? acc.white.data() : acc.black.data();

            // Remove moved piece from old square (AVX2: 8 floats per op)
            int oldFeature = halfKAv2Feature(kingSq, movedPiece, movedColor, fromSq, perspective);
            if (oldFeature >= 0) {
                const float* w = (*L1_weights)[oldFeature].data();
                for (int j = 0; j < L1_SIZE; j += 8) {
                    __m256 a = _mm256_loadu_ps(&accumulator[j]);
                    __m256 wv = _mm256_loadu_ps(&w[j]);
                    _mm256_storeu_ps(&accumulator[j], _mm256_sub_ps(a, wv));
                }
            }

            // Remove captured piece if any (AVX2)
            if (capturedPiece != PieceType::None) {
                int capFeature = halfKAv2Feature(kingSq, capturedPiece, capturedColor, toSq, perspective);
                if (capFeature >= 0) {
                    const float* w = (*L1_weights)[capFeature].data();
                    for (int j = 0; j < L1_SIZE; j += 8) {
                        __m256 a = _mm256_loadu_ps(&accumulator[j]);
                        __m256 wv = _mm256_loadu_ps(&w[j]);
                        _mm256_storeu_ps(&accumulator[j], _mm256_sub_ps(a, wv));
                    }
                }
            }

            // Add moved piece at new square (AVX2)
            int newFeature = halfKAv2Feature(kingSq, movedPiece, movedColor, toSq, perspective);
            if (newFeature >= 0) {
                const float* w = (*L1_weights)[newFeature].data();
                for (int j = 0; j < L1_SIZE; j += 8) {
                    __m256 a = _mm256_loadu_ps(&accumulator[j]);
                    __m256 wv = _mm256_loadu_ps(&w[j]);
                    _mm256_storeu_ps(&accumulator[j], _mm256_add_ps(a, wv));
                }
            }
        }
    }

    // =========================================================================
    // addFeature / removeFeature: incremental single-feature accumulator update
    // NOTE (M-2): These apply the same weight row to BOTH perspectives.
    //   This is only correct when used after a full refresh (computeAccumulator).
    //   For proper HalfKAv2 incremental updates, separate per-perspective
    //   feature indices are needed. Currently the engine always calls
    //   computeAccumulator, so this works, but callers should be aware.
    // =========================================================================
    // M-D2 FIX: Perspective-aware addFeature/removeFeature for correct HalfKAv2 incremental updates.
    // Each perspective gets its own feature index (computed via halfKAv2Feature with the
    // appropriate king square and perspective color).
    void Network::addFeature(int whiteFeatureIdx, int blackFeatureIdx, Accumulator& acc) const {
        if (!L1_weights) return;  // FIX 2.19: guard against null after releaseFloatWeights
        if (whiteFeatureIdx >= 0 && whiteFeatureIdx < NUM_FEATURES) {
            const float* w = (*L1_weights)[whiteFeatureIdx].data();
            float* a = acc.white.data();
            for (int i = 0; i < L1_SIZE; i += 8) {
                __m256 av = _mm256_loadu_ps(&a[i]);
                __m256 wv = _mm256_loadu_ps(&w[i]);
                _mm256_storeu_ps(&a[i], _mm256_add_ps(av, wv));
            }
        }
        if (blackFeatureIdx >= 0 && blackFeatureIdx < NUM_FEATURES) {
            const float* w = (*L1_weights)[blackFeatureIdx].data();
            float* a = acc.black.data();
            for (int i = 0; i < L1_SIZE; i += 8) {
                __m256 av = _mm256_loadu_ps(&a[i]);
                __m256 wv = _mm256_loadu_ps(&w[i]);
                _mm256_storeu_ps(&a[i], _mm256_add_ps(av, wv));
            }
        }
    }

    void Network::removeFeature(int whiteFeatureIdx, int blackFeatureIdx, Accumulator& acc) const {
        if (!L1_weights) return;  // FIX 2.19: guard against null after releaseFloatWeights
        if (whiteFeatureIdx >= 0 && whiteFeatureIdx < NUM_FEATURES) {
            const float* w = (*L1_weights)[whiteFeatureIdx].data();
            float* a = acc.white.data();
            for (int i = 0; i < L1_SIZE; i += 8) {
                __m256 av = _mm256_loadu_ps(&a[i]);
                __m256 wv = _mm256_loadu_ps(&w[i]);
                _mm256_storeu_ps(&a[i], _mm256_sub_ps(av, wv));
            }
        }
        if (blackFeatureIdx >= 0 && blackFeatureIdx < NUM_FEATURES) {
            const float* w = (*L1_weights)[blackFeatureIdx].data();
            float* a = acc.black.data();
            for (int i = 0; i < L1_SIZE; i += 8) {
                __m256 av = _mm256_loadu_ps(&a[i]);
                __m256 wv = _mm256_loadu_ps(&w[i]);
                _mm256_storeu_ps(&a[i], _mm256_sub_ps(av, wv));
            }
        }
    }

    // Legacy single-index overloads (applies same weights to both perspectives).
    // WARNING: Only correct immediately after a full refreshAccumulator() call.
    // For proper incremental updates, use the two-index overload above.
    void Network::addFeature(int featureIdx, Accumulator& acc) const {
        addFeature(featureIdx, featureIdx, acc);
    }

    void Network::removeFeature(int featureIdx, Accumulator& acc) const {
        removeFeature(featureIdx, featureIdx, acc);
    }

    // =========================================================================
    // forward: backward-compatible overload (defaults to phase=0.5)
    // =========================================================================
    int Network::forward(const Accumulator& acc, Color sideToMove) const {
        return forward(acc, sideToMove, 0.5f);
    }

    // =========================================================================
    // forward: SIMD-optimized forward pass with SCReLU activation and phase heads
    // =========================================================================
    int Network::forward(const Accumulator& acc, Color sideToMove, float phase) const {
        // Build input with SCReLU — AVX2 accelerated
        alignas(32) float input[L1_SIZE * 2]{};

        const auto& stmAcc = (sideToMove == Color::White) ? acc.white : acc.black;
        const auto& oppAcc = (sideToMove == Color::White) ? acc.black : acc.white;

        // SCReLU on STM accumulator (AVX2: 8 floats at a time)
        for (int i = 0; i < L1_SIZE; i += 8) {
            __m256 val = _mm256_loadu_ps(&stmAcc[i]);
            _mm256_storeu_ps(&input[i], screlu_avx(val));
        }
        // SCReLU on opponent accumulator (AVX2)
        for (int i = 0; i < L1_SIZE; i += 8) {
            __m256 val = _mm256_loadu_ps(&oppAcc[i]);
            _mm256_storeu_ps(&input[L1_SIZE + i], screlu_avx(val));
        }

        // L2: 1024 -> 128 (AVX2 dot products)
        alignas(32) float l2Out[L2_SIZE]{};
        for (int j = 0; j < L2_SIZE; ++j) {
            __m256 sum = _mm256_setzero_ps();
            for (int i = 0; i < L1_SIZE * 2; i += 8) {
                __m256 inp = _mm256_loadu_ps(&input[i]);
                __m256 w   = _mm256_loadu_ps(&L2_weights_T[j][i]);
                sum = _mm256_fmadd_ps(inp, w, sum);
            }
            float val = hsum_avx(sum) + L2_biases[j];
            float clamped = std::max(0.0f, std::min(val, 1.0f));
            l2Out[j] = clamped * clamped;
        }

        // L3: 128 -> 64 (AVX2 dot products)
        alignas(32) float l3Out[L3_SIZE]{};
        for (int j = 0; j < L3_SIZE; ++j) {
            __m256 sum = _mm256_setzero_ps();
            for (int i = 0; i < L2_SIZE; i += 8) {
                __m256 inp = _mm256_loadu_ps(&l2Out[i]);
                __m256 w   = _mm256_loadu_ps(&L3_weights_T[j][i]);
                sum = _mm256_fmadd_ps(inp, w, sum);
            }
            float val = hsum_avx(sum) + L3_biases[j];
            float clamped = std::max(0.0f, std::min(val, 1.0f));
            l3Out[j] = clamped * clamped;
        }

        // OPT #3: Vectorized phase heads — AVX2 dot products + fast exp approximation
        auto computeWDL = [&](const PhaseHead& head, float wdl[WDL_SIZE]) {
            for (int k = 0; k < WDL_SIZE; ++k) {
                __m256 sum = _mm256_setzero_ps();
                for (int i = 0; i < L3_SIZE; i += 8) {
                    __m256 inp = _mm256_loadu_ps(&l3Out[i]);
                    __m256 w   = _mm256_loadu_ps(&head.weights[k][i]);
                    sum = _mm256_fmadd_ps(inp, w, sum);
                }
                wdl[k] = hsum_avx(sum) + head.biases[k];
            }
            float maxVal = std::max({wdl[0], wdl[1], wdl[2]});
            float expSum = 0.0f;
            for (int k = 0; k < WDL_SIZE; ++k) {
                wdl[k] = fast_expf(wdl[k] - maxVal);
                expSum += wdl[k];
            }
            float invSum = 1.0f / expSum;
            for (int k = 0; k < WDL_SIZE; ++k)
                wdl[k] *= invSum;
        };

        float op_wdl[WDL_SIZE]{}, mg_wdl[WDL_SIZE]{}, eg_wdl[WDL_SIZE]{};
        computeWDL(head_opening, op_wdl);
        computeWDL(head_middlegame, mg_wdl);
        computeWDL(head_endgame, eg_wdl);

        // Phase blend weights using degree-2 Bernstein polynomial basis (sum to 1.0):
        //   w_op = p²          — opening weight, dominant near p=1
        //   w_mg = 2·p·(1−p)   — middlegame weight, peaks at p=0.5
        //   w_eg = (1−p)²      — endgame weight, dominant near p=0
        // At the midpoint p=0.5 the weights are 0.25/0.50/0.25, giving the
        // middlegame head twice the influence of either wing head.
        // 2.22: Bernstein quadratic basis: w_op=p^2, w_mg=2p(1-p), w_eg=(1-p)^2
        float p = phase;
        float w_op = p * p;
        float w_eg = (1.0f - p) * (1.0f - p);
        float w_mg = 2.0f * p * (1.0f - p);

        // Blended WDL
        float wdl[WDL_SIZE]{};
        for (int k = 0; k < WDL_SIZE; ++k) {
            wdl[k] = w_op * op_wdl[k] + w_mg * mg_wdl[k] + w_eg * eg_wdl[k];
        }

        // Score = (win - loss) * 400 centipawns
        float output = (wdl[0] - wdl[2]) * 400.0f;

        return static_cast<int>(std::lround(output));  // 2.17/15.3: round instead of truncate
    }

    void Network::transposeWeights() {
        for (int i = 0; i < L1_SIZE * 2; ++i)
            for (int j = 0; j < L2_SIZE; ++j)
                L2_weights_T[j][i] = L2_weights[i][j];
        for (int i = 0; i < L2_SIZE; ++i)
            for (int j = 0; j < L3_SIZE; ++j)
                L3_weights_T[j][i] = L3_weights[i][j];

        // OPT A: Re-quantize INT8 weights after transposition
        quantizeL2L3Weights();
    }

    // PHASE 3: Quantize float L1 weights/biases to INT16 (scale = QA = 256)
    void Network::quantizeWeights() {
        // 15.7: Defensive re-allocation in case L1_weights_q was ever released
        if (!L1_weights_q)
            L1_weights_q = std::make_unique<std::array<std::array<int16_t, L1_SIZE>, NUM_FEATURES>>();

        // 15.4: Track clamped weights to detect train/eval mismatch
        int clampCount = 0;
        for (int f = 0; f < NUM_FEATURES; ++f) {
            for (int j = 0; j < L1_SIZE; ++j) {
                float w = (*L1_weights)[f][j];
                int q = (int)std::round(w * QA);
                if (q < -32768 || q > 32767) clampCount++;
                // Clamp to int16 range
                q = std::max(-32768, std::min(32767, q));
                (*L1_weights_q)[f][j] = (int16_t)q;
            }
        }
        if (clampCount > 0)
            std::cerr << "NNUE quantization: " << clampCount
                      << " weights clamped to int16 range (train/eval mismatch risk)" << std::endl;

        for (int j = 0; j < L1_SIZE; ++j) {
            int q = (int)std::round(L1_biases[j] * QA);
            q = std::max(-32768, std::min(32767, q));
            L1_biases_q[j] = (int16_t)q;
        }

        // OPT A: Also quantize L2/L3 to INT8
        quantizeL2L3Weights();
    }

    // OPT A: Quantize L2/L3 float weights to INT8 for fast inference
    void Network::quantizeL2L3Weights() {
        // L2 transposed weights: float → int8 (scale = QW_L2)
        int clampL2 = 0;
        for (int j = 0; j < L2_SIZE; ++j) {
            for (int i = 0; i < L1_SIZE * 2; ++i) {
                int q = (int)std::round(L2_weights_T[j][i] * QW_L2);
                if (q < -127 || q > 127) clampL2++;
                q = std::max(-127, std::min(127, q));
                L2_weights_T_q[j][i] = (int8_t)q;
            }
            // Pre-scale bias: bias_q = round(float_bias * QA_ACT * QW_L2)
            L2_biases_q[j] = (int32_t)std::round(L2_biases[j] * QA_ACT * QW_L2);
        }
        if (clampL2 > 0)
            std::cerr << "NNUE L2 INT8 quantization: " << clampL2 << " weights clamped" << std::endl;

        // L3 transposed weights: float → int8 (scale = QW_L3)
        int clampL3 = 0;
        for (int j = 0; j < L3_SIZE; ++j) {
            for (int i = 0; i < L2_SIZE; ++i) {
                int q = (int)std::round(L3_weights_T[j][i] * QW_L3);
                if (q < -127 || q > 127) clampL3++;
                q = std::max(-127, std::min(127, q));
                L3_weights_T_q[j][i] = (int8_t)q;
            }
            L3_biases_q[j] = (int32_t)std::round(L3_biases[j] * QA_ACT * QW_L3);
        }
        if (clampL3 > 0)
            std::cerr << "NNUE L3 INT8 quantization: " << clampL3 << " weights clamped" << std::endl;
    }

    void Network::releaseFloatWeights() {
        L1_weights.reset();  // free ~160 MB heap allocation
    }

    // Magic number and version for weight file format validation
    // Version 5: Stores architecture dimensions in header; automatic weight migration.
    // Version 4: Phase heads + scaled network (L1=1024, L2=128, L3=64). Fixed dims.
    // Version 3: Phase heads, smaller network (L1=512, L2=64, L3=32) — incompatible.
    // Version 2: Single output head. Version 1: legacy 768 features.
    static constexpr uint32_t NNUE_MAGIC   = 0x4E4E5545;  // "NNUE" in ASCII
    static constexpr uint32_t NNUE_VERSION = 5;

    // Helper to read a PhaseHead from a binary stream.
    // WARNING (15.6): The binary weight format stores raw float/int16 values
    // without byte-swapping. Files are NOT portable across architectures with
    // different endianness (e.g., x86 little-endian vs. ARM big-endian).
    static bool readPhaseHead(std::ifstream& file, PhaseHead& head) {
        file.read(reinterpret_cast<char*>(head.weights.data()), sizeof(head.weights));
        file.read(reinterpret_cast<char*>(head.biases.data()), sizeof(head.biases));
        return file.good();
    }

    // Helper to write a PhaseHead to a binary stream
    static bool writePhaseHead(std::ofstream& file, const PhaseHead& head) {
        file.write(reinterpret_cast<const char*>(head.weights.data()), sizeof(head.weights));
        file.write(reinterpret_cast<const char*>(head.biases.data()), sizeof(head.biases));
        return file.good();
    }

    // Helper: initialize a PhaseHead from old single-output weights
    // Copies old output weights into "win" row, initializes draw/loss to small values
    static void initPhaseHeadFromOldOutput(PhaseHead& head, const float* oldWeights, int oldSize, float oldBias) {
        // Zero everything first
        for (auto& row : head.weights)
            row.fill(0.0f);
        head.biases.fill(0.0f);

        // Copy old output weights into win row (index 0), clamped to L3_SIZE
        int copySize = std::min(oldSize, L3_SIZE);
        for (int i = 0; i < copySize; ++i) {
            head.weights[0][i] = oldWeights[i];  // win
        }
        head.biases[0] = oldBias;

        // Loss row gets negated weights (opposing signal)
        for (int i = 0; i < copySize; ++i) {
            head.weights[2][i] = -oldWeights[i];
        }
        head.biases[2] = -oldBias;

        // Draw row stays at zero (neutral)
    }

    // =========================================================================
    // Adaptive weight loading helpers for architecture migration
    // =========================================================================

    // Read a [fileRows x fileCols] matrix from stream into a [targetRows x targetCols]
    // target buffer (which must be pre-zeroed). Copies the overlapping region.
    static void adaptiveRead2D(std::ifstream& file, float* target,
                               uint32_t targetRows, uint32_t targetCols,
                               uint32_t fileRows, uint32_t fileCols) {
        uint32_t copyCols = std::min(fileCols, targetCols);
        std::vector<float> rowBuf(fileCols);
        for (uint32_t r = 0; r < fileRows; r++) {
            file.read(reinterpret_cast<char*>(rowBuf.data()), fileCols * sizeof(float));
            if (r < targetRows) {
                std::memcpy(target + (size_t)r * targetCols, rowBuf.data(),
                            copyCols * sizeof(float));
            }
        }
    }

    // Read a [fileSize] bias vector into a [targetSize] buffer (pre-zeroed).
    static void adaptiveRead1D(std::ifstream& file, float* target,
                               uint32_t targetSize, uint32_t fileSize) {
        uint32_t copySize = std::min(fileSize, targetSize);
        std::vector<float> buf(fileSize);
        file.read(reinterpret_cast<char*>(buf.data()), fileSize * sizeof(float));
        std::memcpy(target, buf.data(), copySize * sizeof(float));
    }

    // He-initialize new columns [colStart, colEnd) for rows [0, rowEnd).
    // Uses the same Glorot-like formula as randomizeWeights(): sqrt(2 / (fanIn + fanOut)).
    // Only initializes incoming weights to new neurons from existing (old) inputs.
    static void heInitCols(float* target, uint32_t stride,
                           uint32_t colStart, uint32_t colEnd, uint32_t rowEnd,
                           uint32_t fanIn, uint32_t fanOut, std::mt19937& rng) {
        float stddev = std::sqrt(2.0f / (fanIn + fanOut));
        std::normal_distribution<float> dist(0.0f, stddev);
        for (uint32_t r = 0; r < rowEnd; r++) {
            for (uint32_t c = colStart; c < colEnd; c++) {
                target[(size_t)r * stride + c] = dist(rng);
            }
        }
    }

    bool Network::loadWeights(const std::string& filename) {
        // Re-allocate L1_weights if previously released (e.g., after releaseFloatWeights)
        if (!L1_weights) {
            L1_weights = std::make_unique<std::array<std::array<float, L1_SIZE>, NUM_FEATURES>>();
        }

        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) return false;

        // Read and validate header
        uint32_t magic = 0, version = 0;
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));

        if (magic != NNUE_MAGIC) {
            // Legacy 768-feature file or garbage — can't migrate
            std::cerr << "NNUE: Legacy weight file incompatible with current architecture. "
                      << "Starting with random weights \u2014 please retrain." << std::endl;
            randomizeWeights();
            transposeWeights();
            quantizeWeights();
            return false;
        }

        file.read(reinterpret_cast<char*>(&version), sizeof(version));

        if (version == 2 || version == 3) {
            // V2/V3: L1=512 era — architecture too different to migrate
            std::cerr << "NNUE: Weight file version " << version
                      << " is incompatible (L1=512 era). "
                      << "Starting with random weights \u2014 please retrain." << std::endl;
            randomizeWeights();
            transposeWeights();
            quantizeWeights();
            return false;
        }

        // ================================================================
        // Determine the file's architecture dimensions
        // ================================================================
        uint32_t fl1 = 0, fl2 = 0, fl3 = 0, fwdl = 0;

        if (version == 5) {
            file.read(reinterpret_cast<char*>(&fl1),  sizeof(fl1));
            file.read(reinterpret_cast<char*>(&fl2),  sizeof(fl2));
            file.read(reinterpret_cast<char*>(&fl3),  sizeof(fl3));
            file.read(reinterpret_cast<char*>(&fwdl), sizeof(fwdl));
        } else if (version == 4) {
            // V4 had fixed dimensions (no header dims)
            fl1 = 1024; fl2 = 128; fl3 = 64; fwdl = 3;
        } else {
            std::cerr << "NNUE: Unknown weight file version " << version << "." << std::endl;
            return false;
        }

        // Sanity-check dimensions to avoid OOM on corrupt files
        if (fl1 == 0 || fl1 > 4096 || fl2 == 0 || fl2 > 2048 ||
            fl3 == 0 || fl3 > 2048 || fwdl == 0 || fwdl > 16) {
            std::cerr << "NNUE: Invalid dimensions in weight file (L1=" << fl1
                      << " L2=" << fl2 << " L3=" << fl3 << " WDL=" << fwdl
                      << "). File may be corrupt." << std::endl;
            return false;
        }

        bool migrating = (fl1 != (uint32_t)L1_SIZE || fl2 != (uint32_t)L2_SIZE ||
                          fl3 != (uint32_t)L3_SIZE || fwdl != (uint32_t)WDL_SIZE);

        if (migrating) {
            // ============================================================
            // Architecture migration: adaptive loading + He-init
            // - Existing neurons: weights copied exactly (behavior preserved)
            // - New neurons: incoming weights = He-init (breaks symmetry),
            //   outgoing weights = zero (no immediate effect on output)
            // - Removed neurons: truncated (optimizer will adapt)
            // ============================================================
            std::cout << "NNUE: Migrating weights from "
                      << NUM_FEATURES << "x" << fl1 << "x" << fl2
                      << "x" << fl3 << "x" << fwdl
                      << " -> " << NUM_FEATURES << "x" << L1_SIZE << "x" << L2_SIZE
                      << "x" << L3_SIZE << "x" << WDL_SIZE << std::endl;

            // Deterministic RNG for reproducible migration
            std::mt19937 rng(12345);

            // Zero all weights first
            for (auto& row : *L1_weights) row.fill(0.0f);
            L1_biases.fill(0.0f);
            for (auto& row : L2_weights) row.fill(0.0f);
            L2_biases.fill(0.0f);
            for (auto& row : L3_weights) row.fill(0.0f);
            L3_biases.fill(0.0f);
            auto zeroHead = [](PhaseHead& h) {
                for (auto& r : h.weights) r.fill(0.0f);
                h.biases.fill(0.0f);
            };
            zeroHead(head_opening);
            zeroHead(head_middlegame);
            zeroHead(head_endgame);

            // --- L1: [NUM_FEATURES x fl1] -> [NUM_FEATURES x L1_SIZE] ---
            adaptiveRead2D(file, &(*L1_weights)[0][0],
                           NUM_FEATURES, L1_SIZE, NUM_FEATURES, fl1);
            adaptiveRead1D(file, L1_biases.data(), L1_SIZE, fl1);
            if (L1_SIZE > (int)fl1) {
                heInitCols(&(*L1_weights)[0][0], L1_SIZE,
                           fl1, L1_SIZE, NUM_FEATURES,
                           NUM_FEATURES, L1_SIZE, rng);
                std::cout << "  L1: expanded " << fl1 << " -> " << L1_SIZE
                          << " neurons (He-init new incoming weights)" << std::endl;
            } else if (L1_SIZE < (int)fl1) {
                std::cout << "  L1: truncated " << fl1 << " -> " << L1_SIZE
                          << " neurons" << std::endl;
            }

            // --- L2: [fl1*2 x fl2] -> [L1_SIZE*2 x L2_SIZE] ---
            // New rows (from L1 expansion) stay zero = correct (outgoing from new L1 neurons)
            adaptiveRead2D(file, &L2_weights[0][0],
                           L1_SIZE * 2, L2_SIZE, fl1 * 2, fl2);
            adaptiveRead1D(file, L2_biases.data(), L2_SIZE, fl2);
            if (L2_SIZE > (int)fl2) {
                // He-init incoming weights to new L2 neurons, only from OLD L1 neurons
                uint32_t oldInputRows = std::min(fl1 * 2, (uint32_t)(L1_SIZE * 2));
                heInitCols(&L2_weights[0][0], L2_SIZE,
                           fl2, L2_SIZE, oldInputRows,
                           L1_SIZE * 2, L2_SIZE, rng);
                std::cout << "  L2: expanded " << fl2 << " -> " << L2_SIZE
                          << " neurons (He-init new incoming weights)" << std::endl;
            } else if (L2_SIZE < (int)fl2) {
                std::cout << "  L2: truncated " << fl2 << " -> " << L2_SIZE
                          << " neurons" << std::endl;
            }

            // --- L3: [fl2 x fl3] -> [L2_SIZE x L3_SIZE] ---
            // New rows (from L2 expansion) stay zero = correct (outgoing from new L2 neurons)
            adaptiveRead2D(file, &L3_weights[0][0],
                           L2_SIZE, L3_SIZE, fl2, fl3);
            adaptiveRead1D(file, L3_biases.data(), L3_SIZE, fl3);
            if (L3_SIZE > (int)fl3) {
                uint32_t oldInputRows = std::min(fl2, (uint32_t)L2_SIZE);
                heInitCols(&L3_weights[0][0], L3_SIZE,
                           fl3, L3_SIZE, oldInputRows,
                           L2_SIZE, L3_SIZE, rng);
                std::cout << "  L3: expanded " << fl3 << " -> " << L3_SIZE
                          << " neurons (He-init new incoming weights)" << std::endl;
            } else if (L3_SIZE < (int)fl3) {
                std::cout << "  L3: truncated " << fl3 << " -> " << L3_SIZE
                          << " neurons" << std::endl;
            }

            // --- Phase heads: [fwdl x fl3] -> [WDL_SIZE x L3_SIZE] ---
            // Outgoing weights from new L3 neurons stay zero (correct)
            auto migrateHead = [&](PhaseHead& head) {
                adaptiveRead2D(file, &head.weights[0][0],
                               WDL_SIZE, L3_SIZE, fwdl, fl3);
                adaptiveRead1D(file, head.biases.data(), WDL_SIZE, fwdl);
            };
            migrateHead(head_opening);
            migrateHead(head_middlegame);
            migrateHead(head_endgame);

            if (!file.good()) {
                std::cerr << "NNUE weight file truncated or corrupt during migration."
                          << std::endl;
                randomizeWeights();
                transposeWeights();
                quantizeWeights();
                return false;
            }

            std::cout << "NNUE: Migration complete. Existing neurons preserved, "
                      << "new neurons initialized with He weights." << std::endl;

        } else {
            // ============================================================
            // Fast path: dimensions match exactly, direct read
            // ============================================================
            file.read(reinterpret_cast<char*>(L1_weights->data()), sizeof(*L1_weights));
            file.read(reinterpret_cast<char*>(L1_biases.data()), sizeof(L1_biases));
            file.read(reinterpret_cast<char*>(L2_weights.data()), sizeof(L2_weights));
            file.read(reinterpret_cast<char*>(L2_biases.data()), sizeof(L2_biases));
            file.read(reinterpret_cast<char*>(L3_weights.data()), sizeof(L3_weights));
            file.read(reinterpret_cast<char*>(L3_biases.data()), sizeof(L3_biases));
            readPhaseHead(file, head_opening);
            readPhaseHead(file, head_middlegame);
            readPhaseHead(file, head_endgame);

            if (!file.good()) {
                std::cerr << "NNUE weight file truncated or corrupt." << std::endl;
                randomizeWeights();
                transposeWeights();
                quantizeWeights();
                return false;
            }
        }

        transposeWeights();
        quantizeWeights();
        return true;
    }

    bool Network::saveWeights(const std::string& filename) {
        if (!L1_weights) {
            std::cerr << "NNUE: Cannot save — float weights released. "
                      << "Call loadWeights() to re-allocate first." << std::endl;
            return false;
        }
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) return false;

        // V5 header: magic + version + architecture dimensions
        // WARNING (15.6): Binary format is endianness-dependent; not portable across architectures.
        uint32_t magic = NNUE_MAGIC;
        uint32_t version = NNUE_VERSION;
        uint32_t dims[4] = {
            (uint32_t)L1_SIZE, (uint32_t)L2_SIZE,
            (uint32_t)L3_SIZE, (uint32_t)WDL_SIZE
        };
        file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));
        file.write(reinterpret_cast<const char*>(dims), sizeof(dims));

        file.write(reinterpret_cast<const char*>(L1_weights->data()), sizeof(*L1_weights));
        file.write(reinterpret_cast<const char*>(L1_biases.data()), sizeof(L1_biases));
        file.write(reinterpret_cast<const char*>(L2_weights.data()), sizeof(L2_weights));
        file.write(reinterpret_cast<const char*>(L2_biases.data()), sizeof(L2_biases));
        file.write(reinterpret_cast<const char*>(L3_weights.data()), sizeof(L3_weights));
        file.write(reinterpret_cast<const char*>(L3_biases.data()), sizeof(L3_biases));

        // Write 3 phase heads
        writePhaseHead(file, head_opening);
        writePhaseHead(file, head_middlegame);
        writePhaseHead(file, head_endgame);

        return file.good();
    }

    void Network::randomizeWeights(float scale, int seed) {
        std::mt19937 rng(seed);

        {
            float stddev = scale * std::sqrt(2.0f / (NUM_FEATURES + L1_SIZE));
            std::normal_distribution<float> dist(0.0f, stddev);
            for (auto& row : *L1_weights)
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

        // Phase heads: He initialization
        {
            float stddev = scale * std::sqrt(2.0f / (L3_SIZE + WDL_SIZE));
            std::normal_distribution<float> dist(0.0f, stddev);

            auto initHead = [&](PhaseHead& head) {
                for (auto& row : head.weights)
                    for (auto& w : row)
                        w = dist(rng);
                head.biases.fill(0.0f);
            };

            initHead(head_opening);
            initHead(head_middlegame);
            initHead(head_endgame);
        }

        transposeWeights();
        quantizeWeights();
    }

    // =========================================================================
    // PHASE 3: Quantized (INT16) accumulator refresh — SSE2 with 8 int16 per op
    // Uses saturating adds (_mm_adds_epi16) to prevent silent overflow.
    // 2x throughput vs float path due to int16 packing (8 ops per 128-bit SSE).
    // =========================================================================
    void Network::refreshAccumulatorQ(const Board& board, QAccumulator& acc) const {
        // Safety guard — L1_weights_q must always be initialized
        if (!L1_weights_q) {
            // Fallback: zero accumulator (will produce wrong eval but won't crash)
            acc.white.fill(0); acc.black.fill(0);
            acc.valid = false; return;
        }
        // Use Board's cached king squares (avoids 64-square mailbox scan)
        int wKingSq = squareIndex(board.whiteKingSq.rank, board.whiteKingSq.col);
        int bKingSq = squareIndex(board.blackKingSq.rank, board.blackKingSq.col);
        acc.whiteKingSq = wKingSq;
        acc.blackKingSq = bKingSq;

        // Initialize with quantized biases (AVX2: 16 int16 per op — 2x throughput vs SSE2)
        for (int j = 0; j < L1_SIZE; j += 16) {
            __m256i bias = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&L1_biases_q[j]));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&acc.white[j]), bias);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&acc.black[j]), bias);
        }

        // M-D1 FIX: Use saturating adds/subs to prevent silent INT16 overflow
        // AVX2: 16 int16 per op (doubled from SSE2's 8)
        // FIX 2.20: Iterate occupied squares via popLsb (O(pieces) vs O(64) mailbox scan)
        {
            Bitboard occ = board.occupied();
            while (occ) {
                int sq = popLsb(occ);
                int rank = sq / 8;
                int col = sq % 8;
                Piece piece = board.squares[rank][col];
                if (piece.isNone() || piece.isDuck() || piece.type == PieceType::King)
                    continue;

                int wFeature = halfKAv2Feature(wKingSq, piece.type, piece.color, sq, Color::White);
                if (wFeature >= 0) {
                    const int16_t* w = (*L1_weights_q)[wFeature].data();
                    int16_t* a = acc.white.data();
                    for (int j = 0; j < L1_SIZE; j += 16) {
                        __m256i av = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a[j]));
                        __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&w[j]));
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&a[j]), _mm256_adds_epi16(av, wv));
                    }
                }

                int bFeature = halfKAv2Feature(bKingSq, piece.type, piece.color, sq, Color::Black);
                if (bFeature >= 0) {
                    const int16_t* w = (*L1_weights_q)[bFeature].data();
                    int16_t* a = acc.black.data();
                    for (int j = 0; j < L1_SIZE; j += 16) {
                        __m256i av = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a[j]));
                        __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&w[j]));
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&a[j]), _mm256_adds_epi16(av, wv));
                    }
                }
            }
        }
        acc.valid = true;
#ifndef NDEBUG
        // 15.5: Debug check for int16 saturation (silent accuracy loss if triggered)
        for (int j = 0; j < L1_SIZE; ++j) {
            if (acc.white[j] == 32767 || acc.white[j] == -32768 ||
                acc.black[j] == 32767 || acc.black[j] == -32768) {
                std::cerr << "NNUE WARNING: int16 saturation in QAccumulator at j=" << j << std::endl;
                break;
            }
        }
#endif
    }

    // =========================================================================
    // OPT #1: Finny-table-accelerated refresh
    // Instead of rebuilding the accumulator from scratch on king moves,
    // delta-update from the cached state for this king bucket.
    // Typical king move changes 0-3 pieces vs. the cached snapshot,
    // saving ~90% of the work compared to a full refresh (~15 piece adds).
    // =========================================================================
    void Network::refreshAccumulatorQFinny(const Board& board, QAccumulator& acc,
                                            FinnyTable& finny) const {
        int wKingSq = squareIndex(board.whiteKingSq.rank, board.whiteKingSq.col);
        int bKingSq = squareIndex(board.blackKingSq.rank, board.blackKingSq.col);
        acc.whiteKingSq = wKingSq;
        acc.blackKingSq = bKingSq;

        for (int persp = 0; persp < 2; ++persp) {
            Color perspective = static_cast<Color>(persp);
            int kingSq = (persp == 0) ? wKingSq : bKingSq;
            int16_t* accData = (persp == 0) ? acc.white.data() : acc.black.data();
            FinnyEntry& entry = finny.entries[persp][kingSq];

            if (entry.valid) {
                // Delta update from cached state
                // Start with cached accumulator values
                std::memcpy(accData, entry.values.data(), L1_SIZE * sizeof(int16_t));

                // For each non-king piece type and color, find added/removed pieces
                for (int c = 0; c < 2; ++c) {
                    Color pieceColor = static_cast<Color>(c);
                    for (int pt = 1; pt <= 5; ++pt) {  // Pawn=1..Queen=5
                        PieceType pieceType = static_cast<PieceType>(pt);
                        Bitboard current = board.colorBB[c] & board.pieceBBs[pt];
                        Bitboard cached  = entry.pieces[c][pt];

                        // Removed pieces: were in cache but not on board now
                        Bitboard removed = cached & ~current;
                        while (removed) {
                            int sq = popLsb(removed);
                            int feat = halfKAv2Feature(kingSq, pieceType, pieceColor, sq, perspective);
                            if (feat >= 0) {
                                const int16_t* w = (*L1_weights_q)[feat].data();
                                for (int j = 0; j < L1_SIZE; j += 16) {
                                    __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&accData[j]));
                                    __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&w[j]));
                                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(&accData[j]),
                                                       _mm256_subs_epi16(a, wv));
                                }
                            }
                        }

                        // Added pieces: on board now but not in cache
                        Bitboard added = current & ~cached;
                        while (added) {
                            int sq = popLsb(added);
                            int feat = halfKAv2Feature(kingSq, pieceType, pieceColor, sq, perspective);
                            if (feat >= 0) {
                                const int16_t* w = (*L1_weights_q)[feat].data();
                                for (int j = 0; j < L1_SIZE; j += 16) {
                                    __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&accData[j]));
                                    __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&w[j]));
                                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(&accData[j]),
                                                       _mm256_adds_epi16(a, wv));
                                }
                            }
                        }
                    }
                }
            } else {
                // No cached entry — full refresh for this perspective
                // Initialize with biases
                for (int j = 0; j < L1_SIZE; j += 16) {
                    __m256i bias = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&L1_biases_q[j]));
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(&accData[j]), bias);
                }

                Bitboard occ = board.occupied();
                while (occ) {
                    int sq = popLsb(occ);
                    int rank = sq / 8;
                    int col = sq % 8;
                    Piece piece = board.squares[rank][col];
                    if (piece.isNone() || piece.isDuck() || piece.type == PieceType::King)
                        continue;

                    int feat = halfKAv2Feature(kingSq, piece.type, piece.color, sq, perspective);
                    if (feat >= 0) {
                        const int16_t* w = (*L1_weights_q)[feat].data();
                        for (int j = 0; j < L1_SIZE; j += 16) {
                            __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&accData[j]));
                            __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&w[j]));
                            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&accData[j]),
                                               _mm256_adds_epi16(a, wv));
                        }
                    }
                }
            }

            // Update cache entry
            std::memcpy(entry.values.data(), accData, L1_SIZE * sizeof(int16_t));
            for (int c = 0; c < 2; ++c) {
                for (int pt = 1; pt <= 5; ++pt) {
                    entry.pieces[c][pt] = board.colorBB[c] & board.pieceBBs[pt];
                }
            }
            entry.valid = true;
        }
        acc.valid = true;
    }

    // =========================================================================
    // PHASE 3: Quantized (INT16) incremental accumulator update
    // M-D1 FIX: Uses saturating adds/subs to prevent silent INT16 overflow.
    // Key optimization: int16 add/sub per feature change is the hot path in
    // search — runs once per move per ply. 2x throughput vs float with SSE2.
    // =========================================================================
    void Network::incrementalUpdateQ(const Board& board, QAccumulator& acc,
                                     int fromRank, int fromCol, int toRank, int toCol,
                                     PieceType movedPiece, Color movedColor,
                                     PieceType capturedPiece, Color capturedColor) const {
#ifndef NDEBUG
        // Guard: incrementalUpdateQ cannot handle promotions or en passant.
        // Callers must use refreshAccumulatorQ for these special moves.
        assert(!(movedPiece == PieceType::Pawn && (toRank == 0 || toRank == 7))
            && "incrementalUpdateQ called for promotion — use refreshAccumulatorQ");
        if (movedPiece == PieceType::Pawn && capturedPiece == PieceType::Pawn) {
            assert(fromRank != toRank - 1 || fromCol == toCol || board.getPiece(toRank, toCol).type != PieceType::None
                && "incrementalUpdateQ called for en passant — use refreshAccumulatorQ");
        }
#endif
        // FIX C-2: Auto-refresh accumulator on king moves (all HalfKAv2 features change)
        if (movedPiece == PieceType::King) {
            refreshAccumulatorQ(board, acc);
            return;
        }

        int fromSq = squareIndex(fromRank, fromCol);
        int toSq = squareIndex(toRank, toCol);

        for (int persp = 0; persp < 2; ++persp) {
            Color perspective = static_cast<Color>(persp);
            int kingSq = (perspective == Color::White) ? acc.whiteKingSq : acc.blackKingSq;
            int16_t* accumulator = (perspective == Color::White) ? acc.white.data() : acc.black.data();

            // Remove from old square (AVX2: 16 int16 per op)
            int oldFeature = halfKAv2Feature(kingSq, movedPiece, movedColor, fromSq, perspective);
            if (oldFeature >= 0) {
                const int16_t* w = (*L1_weights_q)[oldFeature].data();
                for (int j = 0; j < L1_SIZE; j += 16) {
                    __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&accumulator[j]));
                    __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&w[j]));
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(&accumulator[j]), _mm256_subs_epi16(a, wv));
                }
            }

            // Remove captured piece (AVX2)
            if (capturedPiece != PieceType::None) {
                int capFeature = halfKAv2Feature(kingSq, capturedPiece, capturedColor, toSq, perspective);
                if (capFeature >= 0) {
                    const int16_t* w = (*L1_weights_q)[capFeature].data();
                    for (int j = 0; j < L1_SIZE; j += 16) {
                        __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&accumulator[j]));
                        __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&w[j]));
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&accumulator[j]), _mm256_subs_epi16(a, wv));
                    }
                }
            }

            // Add to new square (AVX2)
            int newFeature = halfKAv2Feature(kingSq, movedPiece, movedColor, toSq, perspective);
            if (newFeature >= 0) {
                const int16_t* w = (*L1_weights_q)[newFeature].data();
                for (int j = 0; j < L1_SIZE; j += 16) {
                    __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&accumulator[j]));
                    __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&w[j]));
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(&accumulator[j]), _mm256_adds_epi16(a, wv));
                }
            }
        }
    }

    // =========================================================================
    // TIER-1 FIX #5: Fused copy + incremental update
    // Instead of: memcpy(child, parent, 4KB) + incrementalUpdateQ(child)
    // This reads parent, applies +/- weight deltas, writes to child in ONE pass.
    // Reduces memory traffic from 3 passes (read parent + write child + read/modify child)
    // to 2 passes (read parent + write child). ~33% less memory bandwidth.
    // =========================================================================
    void Network::fusedCopyAndUpdateQ(const Board& board,
                                      const QAccumulator& parent, QAccumulator& child,
                                      int fromRank, int fromCol, int toRank, int toCol,
                                      PieceType movedPiece, Color movedColor,
                                      PieceType capturedPiece, Color capturedColor) const {
        // FIX H-1: Runtime guard — en passant and promotions cannot be handled
        // incrementally (EP captured pawn is not on toSq; promotion changes piece type).
        // Fall back to full refresh instead of silently corrupting the accumulator.
        bool isPromotion = (movedPiece == PieceType::Pawn && (toRank == 0 || toRank == 7));
        bool isEnPassant = (movedPiece == PieceType::Pawn && fromCol != toCol
                            && capturedPiece == PieceType::None);
        if (isPromotion || isEnPassant) {
            child.whiteKingSq = parent.whiteKingSq;
            child.blackKingSq = parent.blackKingSq;
            child.valid = parent.valid;
            refreshAccumulatorQ(board, child);
            return;
        }

        // King moves require full refresh (all HalfKAv2 features change)
        if (movedPiece == PieceType::King) {
            child.whiteKingSq = parent.whiteKingSq;
            child.blackKingSq = parent.blackKingSq;
            child.valid = parent.valid;
            refreshAccumulatorQ(board, child);
            return;
        }

        int fromSq = squareIndex(fromRank, fromCol);
        int toSq = squareIndex(toRank, toCol);

        // Copy non-accumulator fields
        child.valid = parent.valid;
        child.whiteKingSq = parent.whiteKingSq;
        child.blackKingSq = parent.blackKingSq;

        for (int persp = 0; persp < 2; ++persp) {
            Color perspective = static_cast<Color>(persp);
            int kingSq = (perspective == Color::White) ? parent.whiteKingSq : parent.blackKingSq;
            const int16_t* src = (perspective == Color::White) ? parent.white.data() : parent.black.data();
            int16_t* dst = (perspective == Color::White) ? child.white.data() : child.black.data();

            // Compute feature indices for delta application
            int oldFeature = halfKAv2Feature(kingSq, movedPiece, movedColor, fromSq, perspective);
            int capFeature = (capturedPiece != PieceType::None)
                ? halfKAv2Feature(kingSq, capturedPiece, capturedColor, toSq, perspective)
                : -1;
            int newFeature = halfKAv2Feature(kingSq, movedPiece, movedColor, toSq, perspective);

            const int16_t* wOld = (oldFeature >= 0) ? (*L1_weights_q)[oldFeature].data() : nullptr;
            const int16_t* wCap = (capFeature >= 0) ? (*L1_weights_q)[capFeature].data() : nullptr;
            const int16_t* wNew = (newFeature >= 0) ? (*L1_weights_q)[newFeature].data() : nullptr;

            // Fused loop: read parent + apply deltas + write child (AVX2, 16 int16 per op)
            for (int j = 0; j < L1_SIZE; j += 16) {
                __m256i val = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&src[j]));

                // Subtract old feature weights (piece removed from old square)
                if (wOld) {
                    __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&wOld[j]));
                    val = _mm256_subs_epi16(val, wv);
                }

                // Subtract captured piece weights
                if (wCap) {
                    __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&wCap[j]));
                    val = _mm256_subs_epi16(val, wv);
                }

                // Add new feature weights (piece placed on new square)
                if (wNew) {
                    __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&wNew[j]));
                    val = _mm256_adds_epi16(val, wv);
                }

                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&dst[j]), val);
            }
        }
    }

    // =========================================================================
    // PHASE 3: forwardQ — forward pass with INT16 accumulator dequantization + SCReLU
    // Strategy: keep accumulator in int16 for fast incremental updates (the hot
    // path), dequantize to float only at inference time for SCReLU + L2/L3.
    // Dequantization uses AVX2: int16→int32→float, then scale by 1/QA.
    // =========================================================================
    int Network::forwardQ(const QAccumulator& acc, Color sideToMove) const {
        return forwardQ(acc, sideToMove, 0.5f);
    }

    int Network::forwardQ(const QAccumulator& acc, Color sideToMove, float phase) const {
        // OPT A: INT8 L2/L3 inference — 4× throughput via _mm256_maddubs_epi16
        
        // Step 1: Dequantize INT16 accumulator → float → SCReLU → quantize to UINT8
        // SCReLU output ∈ [0,1], scaled to [0, QA_ACT=127] as uint8
        alignas(32) uint8_t input_q[L1_SIZE * 2]{};

        const auto& stmAcc = (sideToMove == Color::White) ? acc.white : acc.black;
        const auto& oppAcc = (sideToMove == Color::White) ? acc.black : acc.white;

        const float dequantScale = 1.0f / QA;
        __m256 scale8 = _mm256_set1_ps(dequantScale);
        __m256 qa_scale = _mm256_set1_ps((float)QA_ACT);
        __m256 zero_f = _mm256_setzero_ps();
        __m256 one_f  = _mm256_set1_ps(1.0f);

        // STM: dequantize → SCReLU → quantize to uint8
        for (int i = 0; i < L1_SIZE; i += 8) {
            __m128i vals = _mm_load_si128(reinterpret_cast<const __m128i*>(&stmAcc[i]));
            __m256i i32  = _mm256_cvtepi16_epi32(vals);
            __m256  flt  = _mm256_mul_ps(_mm256_cvtepi32_ps(i32), scale8);
            // SCReLU: clamp [0,1] then square
            __m256 clamped = _mm256_min_ps(_mm256_max_ps(flt, zero_f), one_f);
            __m256 screlu = _mm256_mul_ps(clamped, clamped);
            // Quantize to uint8: round(screlu * QA_ACT)
            __m256i qi = _mm256_cvtps_epi32(_mm256_mul_ps(screlu, qa_scale));
            // Pack int32 → uint8 via scalar store (only 8 values per iteration)
            alignas(32) int32_t tmp[8];
            _mm256_storeu_si256((__m256i*)tmp, qi);
            for (int k = 0; k < 8; ++k)
                input_q[i + k] = (uint8_t)std::min(127, std::max(0, tmp[k]));
        }

        // Opponent: same treatment
        for (int i = 0; i < L1_SIZE; i += 8) {
            __m128i vals = _mm_load_si128(reinterpret_cast<const __m128i*>(&oppAcc[i]));
            __m256i i32  = _mm256_cvtepi16_epi32(vals);
            __m256  flt  = _mm256_mul_ps(_mm256_cvtepi32_ps(i32), scale8);
            __m256 clamped = _mm256_min_ps(_mm256_max_ps(flt, zero_f), one_f);
            __m256 screlu = _mm256_mul_ps(clamped, clamped);
            __m256i qi = _mm256_cvtps_epi32(_mm256_mul_ps(screlu, qa_scale));
            alignas(32) int32_t tmp[8];
            _mm256_storeu_si256((__m256i*)tmp, qi);
            for (int k = 0; k < 8; ++k)
                input_q[L1_SIZE + i + k] = (uint8_t)std::min(127, std::max(0, tmp[k]));
        }

        // Step 2: L2 matmul via INT8 dot products (1024 → 128)
        // _mm256_maddubs_epi16: 32 uint8 × 32 int8 → 16 int16 (adjacent pairs summed)
        // _mm256_madd_epi16 with ones: 16 int16 → 8 int32 (adjacent pairs summed)
        const __m256i ones_16 = _mm256_set1_epi16(1);
        alignas(32) uint8_t l2Out_q[L2_SIZE]{};
        
        // Pre-compute inverse scales for dequantization
        const float l2_inv_scale = 1.0f / (float)(QA_ACT * QW_L2);

        for (int j = 0; j < L2_SIZE; ++j) {
            __m256i acc32 = _mm256_setzero_si256();
            for (int i = 0; i < L1_SIZE * 2; i += 32) {
                __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&input_q[i]));
                __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&L2_weights_T_q[j][i]));
                __m256i prod16 = _mm256_maddubs_epi16(a, b);  // u8×s8 → i16
                acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(prod16, ones_16));  // i16 → i32
            }
            int32_t sum = hsum_epi32(acc32) + L2_biases_q[j];
            float val = (float)sum * l2_inv_scale;
            // SCReLU + quantize to uint8
            float clamped = std::max(0.0f, std::min(val, 1.0f));
            float sq = clamped * clamped;
            l2Out_q[j] = (uint8_t)std::min(127, (int)std::round(sq * QA_ACT));
        }

        // Step 3: L3 matmul via INT8 dot products (128 → 64)
        alignas(32) float l3Out[L3_SIZE]{};
        const float l3_inv_scale = 1.0f / (float)(QA_ACT * QW_L3);

        for (int j = 0; j < L3_SIZE; ++j) {
            __m256i acc32 = _mm256_setzero_si256();
            // L2_SIZE = 128, process 32 at a time (4 iterations)
            for (int i = 0; i < L2_SIZE; i += 32) {
                __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&l2Out_q[i]));
                __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&L3_weights_T_q[j][i]));
                __m256i prod16 = _mm256_maddubs_epi16(a, b);
                acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(prod16, ones_16));
            }
            int32_t sum = hsum_epi32(acc32) + L3_biases_q[j];
            float val = (float)sum * l3_inv_scale;
            float clamped = std::max(0.0f, std::min(val, 1.0f));
            l3Out[j] = clamped * clamped;  // SCReLU — keep as float for phase heads
        }

        // Step 4: Phase heads — remain float (tiny, 3 × 64 dot products)
        auto computeWDL = [&](const PhaseHead& head, float wdl[WDL_SIZE]) {
            for (int k = 0; k < WDL_SIZE; ++k) {
                __m256 sum = _mm256_setzero_ps();
                for (int i = 0; i < L3_SIZE; i += 8) {
                    __m256 inp = _mm256_loadu_ps(&l3Out[i]);
                    __m256 w   = _mm256_loadu_ps(&head.weights[k][i]);
                    sum = _mm256_fmadd_ps(inp, w, sum);
                }
                wdl[k] = hsum_avx(sum) + head.biases[k];
            }
            float maxVal = std::max({wdl[0], wdl[1], wdl[2]});
            float expSum = 0.0f;
            for (int k = 0; k < WDL_SIZE; ++k) {
                wdl[k] = fast_expf(wdl[k] - maxVal);
                expSum += wdl[k];
            }
            float invSum = 1.0f / expSum;
            for (int k = 0; k < WDL_SIZE; ++k)
                wdl[k] *= invSum;
        };

        float op_wdl[WDL_SIZE]{}, mg_wdl[WDL_SIZE]{}, eg_wdl[WDL_SIZE]{};
        computeWDL(head_opening, op_wdl);
        computeWDL(head_middlegame, mg_wdl);
        computeWDL(head_endgame, eg_wdl);

        float p = phase;
        float w_op = p * p;
        float w_eg = (1.0f - p) * (1.0f - p);
        float w_mg = 2.0f * p * (1.0f - p);

        float wdl[WDL_SIZE]{};
        for (int k = 0; k < WDL_SIZE; ++k)
            wdl[k] = w_op * op_wdl[k] + w_mg * mg_wdl[k] + w_eg * eg_wdl[k];

        float output = (wdl[0] - wdl[2]) * 400.0f;
        return static_cast<int>(std::lround(output));
    }

    int Network::evaluateQ(const Board& board) const {
        QAccumulator acc;
        refreshAccumulatorQ(board, acc);
        float phase = computePhase(board);
        return forwardQ(acc, board.turn, phase);
    }

} // namespace NNUE

