#pragma once
// TrainAVX.h — AVX2 primitives for NNUE training forward/backward passes.
// All functions assume:
//   - L1_SIZE = 512  (multiples of 8 floats = one AVX2 register)
//   - L2_SIZE = 128  (multiple of 8)
//   - L3_SIZE = 64   (multiple of 8)
//   - Arrays are at least 32-byte aligned where noted
// Build flags: /arch:AVX2 (MSVC) or -mavx2 (GCC/Clang) — already set in project.

#ifdef _MSC_VER
#  include <intrin.h>
#else
#  include <immintrin.h>
#endif
#include <cstdint>
#include <cstring>
#include <cmath>

namespace TrainAVX {

// ── Scalar fallback flag ──────────────────────────────────────────────────────
// MSVC defines __AVX2__ when /arch:AVX2 is set. GCC/Clang also define it with -mavx2.
#if defined(__AVX2__)
#  define TRAIN_HAS_AVX2 1
#else
#  define TRAIN_HAS_AVX2 0
#endif

// ── SCReLU helpers (scalar, used in single-element contexts) ──────────────────
inline float screlu(float x) {
    float c = x < 0.f ? 0.f : (x > 1.f ? 1.f : x);
    return c * c;
}
inline float screlu_deriv(float x) {
    if (x <= 0.f || x >= 1.f) return 0.f;
    return 2.f * x;
}

// ─────────────────────────────────────────────────────────────────────────────
// avx_add_row: acc[0..N) += src[0..N)    (N must be multiple of 8)
// Used for L1 accumulator update: acc += L1_weights[feat]
// ─────────────────────────────────────────────────────────────────────────────
template<int N>
inline void avx_add_row(float* __restrict acc, const float* __restrict src) {
#if TRAIN_HAS_AVX2
    static_assert(N % 8 == 0, "N must be multiple of 8");
    for (int i = 0; i < N; i += 8) {
        __m256 a = _mm256_loadu_ps(acc + i);
        __m256 b = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(acc + i, _mm256_add_ps(a, b));
    }
#else
    for (int i = 0; i < N; ++i) acc[i] += src[i];
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// avx_axpy: acc[0..N) += scale * src[0..N)
// Used in backward pass: tGrads[feat*L1..] += dL * L1_weights[feat]
// ─────────────────────────────────────────────────────────────────────────────
template<int N>
inline void avx_axpy(float* __restrict acc, const float* __restrict src, float scale) {
#if TRAIN_HAS_AVX2
    static_assert(N % 8 == 0, "N must be multiple of 8");
    __m256 vs = _mm256_set1_ps(scale);
    for (int i = 0; i < N; i += 8) {
        __m256 a = _mm256_loadu_ps(acc + i);
        __m256 b = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(acc + i, _mm256_fmadd_ps(b, vs, a));
    }
#else
    for (int i = 0; i < N; ++i) acc[i] += scale * src[i];
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// avx_screlu: out[i] = clamp(x[i], 0, 1)^2,  pre[i] = x[i]   (N multiple of 8)
// ─────────────────────────────────────────────────────────────────────────────
template<int N>
inline void avx_screlu(const float* __restrict x,
                        float* __restrict pre,
                        float* __restrict out) {
#if TRAIN_HAS_AVX2
    static_assert(N % 8 == 0, "N must be multiple of 8");
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one  = _mm256_set1_ps(1.f);
    for (int i = 0; i < N; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(pre + i, v);
        __m256 c = _mm256_min_ps(_mm256_max_ps(v, zero), one);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(c, c));
    }
#else
    for (int i = 0; i < N; ++i) {
        pre[i] = x[i];
        float c = x[i] < 0.f ? 0.f : (x[i] > 1.f ? 1.f : x[i]);
        out[i] = c * c;
    }
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// avx_screlu_deriv_mul: dst[i] = src[i] * (pre[i] in (0,1) ? 2*pre[i] : 0)
// Used in backward: dLdPre = dLdOut * screlu'(pre)
// ─────────────────────────────────────────────────────────────────────────────
template<int N>
inline void avx_screlu_deriv_mul(const float* __restrict dLdOut,
                                  const float* __restrict pre,
                                  float* __restrict dLdPre) {
#if TRAIN_HAS_AVX2
    static_assert(N % 8 == 0, "N must be multiple of 8");
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one  = _mm256_set1_ps(1.f);
    const __m256 two  = _mm256_set1_ps(2.f);
    for (int i = 0; i < N; i += 8) {
        __m256 p  = _mm256_loadu_ps(pre + i);
        __m256 d  = _mm256_loadu_ps(dLdOut + i);
        // mask: 0 < p < 1
        __m256 gt0 = _mm256_cmp_ps(p, zero, _CMP_GT_OQ);
        __m256 lt1 = _mm256_cmp_ps(p, one,  _CMP_LT_OQ);
        __m256 mask = _mm256_and_ps(gt0, lt1);
        __m256 deriv = _mm256_and_ps(_mm256_mul_ps(two, p), mask);
        _mm256_storeu_ps(dLdPre + i, _mm256_mul_ps(d, deriv));
    }
#else
    for (int i = 0; i < N; ++i) {
        float d = (pre[i] > 0.f && pre[i] < 1.f) ? 2.f * pre[i] : 0.f;
        dLdPre[i] = dLdOut[i] * d;
    }
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// avx_gemv_T: out[j] = bias[j] + sum_i( mat_T[j][i] * in[i] )
//   mat_T is (OUT x IN), i.e., transposed from (IN x OUT).
//   OUT and IN must be multiples of 8.
//   Equivalent to: out = bias + mat^T * in
// Used for L2/L3 forward pass with transposed weight matrices.
// ─────────────────────────────────────────────────────────────────────────────
template<int OUT, int IN>
inline void avx_gemv_T(const float* __restrict mat_T,  // [OUT][IN]
                        const float* __restrict bias,
                        const float* __restrict in,
                        float* __restrict out) {
#if TRAIN_HAS_AVX2
    static_assert(IN  % 8 == 0, "IN must be multiple of 8");
    static_assert(OUT % 8 == 0, "OUT must be multiple of 8");
    for (int j = 0; j < OUT; ++j) {
        const float* row = mat_T + j * IN;
        __m256 acc = _mm256_setzero_ps();
        for (int i = 0; i < IN; i += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(row + i), _mm256_loadu_ps(in + i), acc);
        // Horizontal sum of 8 floats
        __m128 lo  = _mm256_castps256_ps128(acc);
        __m128 hi  = _mm256_extractf128_ps(acc, 1);
        __m128 sum = _mm_add_ps(lo, hi);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        out[j] = _mm_cvtss_f32(sum) + bias[j];
    }
#else
    for (int j = 0; j < OUT; ++j) {
        float s = bias[j];
        for (int i = 0; i < IN; ++i) s += mat_T[j * IN + i] * in[i];
        out[j] = s;
    }
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// avx_dot: returns sum_i( a[i] * b[i] )   (N multiple of 8)
// Used for output layer forward: rawOut = dot(output_weights, l3Out)
// ─────────────────────────────────────────────────────────────────────────────
template<int N>
inline float avx_dot(const float* __restrict a, const float* __restrict b) {
#if TRAIN_HAS_AVX2
    static_assert(N % 8 == 0, "N must be multiple of 8");
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < N; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
    __m128 lo  = _mm256_castps256_ps128(acc);
    __m128 hi  = _mm256_extractf128_ps(acc, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
#else
    float s = 0.f;
    for (int i = 0; i < N; ++i) s += a[i] * b[i];
    return s;
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// avx_outer_add: grad_mat[i*COLS + j] += a[i] * b[j]    (outer product accumulate)
//   Used for weight gradient: dL/dW_L2 += l1Out (outer) dLdL2Pre
//   Loops: i in [0, ROWS), j in [0, COLS)
//   COLS must be multiple of 8.
// ─────────────────────────────────────────────────────────────────────────────
template<int ROWS, int COLS>
inline void avx_outer_add(float* __restrict grad_mat,     // [ROWS * COLS]
                           const float* __restrict a,     // [ROWS]
                           const float* __restrict b) {   // [COLS]
#if TRAIN_HAS_AVX2
    static_assert(COLS % 8 == 0, "COLS must be multiple of 8");
    for (int i = 0; i < ROWS; ++i) {
        __m256 va = _mm256_set1_ps(a[i]);
        float* row = grad_mat + i * COLS;
        for (int j = 0; j < COLS; j += 8) {
            __m256 g = _mm256_loadu_ps(row + j);
            __m256 bv = _mm256_loadu_ps(b + j);
            _mm256_storeu_ps(row + j, _mm256_fmadd_ps(va, bv, g));
        }
    }
#else
    for (int i = 0; i < ROWS; ++i)
        for (int j = 0; j < COLS; ++j)
            grad_mat[i * COLS + j] += a[i] * b[j];
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// avx_matvec_T_add: out[i] += sum_j( mat[i*COLS + j] * v[j] )
//   Used for backward L2→L1: dLdL1Out[i] += sum_j(dLdL2Pre[j] * L2_weights[i][j])
//   mat is [ROWS x COLS] stored row-major, v is [COLS].
//   COLS must be multiple of 8.
// ─────────────────────────────────────────────────────────────────────────────
template<int ROWS, int COLS>
inline void avx_matvec_T_add(float* __restrict out,         // [ROWS]
                              const float* __restrict mat,   // [ROWS * COLS]
                              const float* __restrict v) {   // [COLS]
#if TRAIN_HAS_AVX2
    static_assert(COLS % 8 == 0, "COLS must be multiple of 8");
    for (int i = 0; i < ROWS; ++i) {
        const float* row = mat + i * COLS;
        __m256 acc = _mm256_setzero_ps();
        for (int j = 0; j < COLS; j += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(row + j), _mm256_loadu_ps(v + j), acc);
        __m128 lo  = _mm256_castps256_ps128(acc);
        __m128 hi  = _mm256_extractf128_ps(acc, 1);
        __m128 sum = _mm_add_ps(lo, hi);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        out[i] += _mm_cvtss_f32(sum);
    }
#else
    for (int i = 0; i < ROWS; ++i) {
        float s = 0.f;
        for (int j = 0; j < COLS; ++j) s += mat[i * COLS + j] * v[j];
        out[i] += s;
    }
#endif
}

} // namespace TrainAVX
