/**
 *
 * Reference ISO-C11 Implementation of LESS.
 *
 * @version 2.0 (February 2025)
 *
 * @author Alessandro Barenghi <alessandro.barenghi@polimi.it>
 * @author Gerardo Pelosi <gerardo.pelosi@polimi.it>
 * @author Duc Tri Nguyen <dnguye69@gmu.edu>
 * @author Floyd Zweydinger <zweydfg8+github@rub.de>
 * @author Luke Beckwith <lbeckwit@gmu.edu>
 *
 * This code is hereby placed in the public domain.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **/

#include <string.h>
#include <stdint.h>
#include <stdio.h>

#include "codes.h"
#include "fq_arith.h"
#include "macro.h"
#include "parameters.h"
#include "transpose.h"

/// swap N_pad bytes in r and s
/// \param r[in]: pointer to the first row
/// \param s[in]: pointer to the second row
static
void swap_rows(FQ_ELEM r[N_pad], 
               FQ_ELEM s[N_pad]){
    vec256_t a, b;
    for(uint32_t i=0; i<N_pad; i+=32) {
         vload256(a, (const vec256_t *)(r + i));
         vload256(b, (const vec256_t *)(s + i));
         vstore256((vec256_t *)(r + i), b);
         vstore256((vec256_t *)(s + i), a);
    }
} /* end swap_rows */

/// Calculate pivot flag array
/// \param G[in]: generator matrix in compress format
/// \param pivot_flag[out]: array denoting the pivot columns via a 1, everything
///     else is 0
void generator_get_pivot_flags(const rref_generator_mat_t *const G,
                               uint8_t pivot_flag [N]) {
    for (uint64_t i = 0; i < N; i = i + 1) {
        pivot_flag[i] = 1;
    }

    for (uint64_t i = 0; i < K; i = i + 1) {
        pivot_flag[G->column_pos[i]] = 0;
    }
} /* end generator_get_pivot_flags */

/// NOTE: constant time implementation
/// right-multiplies a generator by a monomial
/// \param res[out]: pointer to an uninitialized generator matrix
/// \param G[in]: full (K \times N) generator matrix
/// \param monom[in]: (random) monomial matrix
void generator_monomial_mul(generator_mat_t *res,
                            const generator_mat_t *const G,
                            const monomial_t *const monom) {
    FQ_ELEM tmp1[N_pad*K_pad] __attribute__((aligned(32)));
    FQ_ELEM tmp2[N_pad*K_pad] __attribute__((aligned(32)));
    matrix_transpose(tmp1, (uint8_t *)G->values, K_pad, N_pad, K_pad, N_pad);

    for (uint64_t i = 0; i < N; i++) {
        const __m256i p = _mm256_set1_epi16(monom->coefficients[i]);
        const uint64_t in_off = i * K_pad;
        const uint64_t out_off = monom->permutation[i] * K_pad;
        for (uint64_t j = 0; j+32 <= K_pad; j += 32) {
            const __m128i a = _mm_loadu_si128((const __m128i *)(tmp1 + in_off + j +  0));
            const __m128i b = _mm_loadu_si128((const __m128i *)(tmp1 + in_off + j + 16));
            const __m256i t0 = _mm256_cvtepu8_epi16(a);
            const __m256i t1 = _mm256_cvtepu8_epi16(b);

            const __m256i t = avx_mul_full256(t0, t1, p, p);
            _mm256_store_si256((__m256i *)(tmp2 + out_off + j), t);
        }
    }
    matrix_transpose((uint8_t *)res->values, tmp2, N_pad, K_pad, N_pad, K_pad);
} /* end generator_monomial_mul */

/// \param G[in/out]: generator matrix
/// \param is_pivot_column[out]: N bytes, set to 1 if this column is a pivot 
/// column. NOTE: the length is `N_pad`, but the upper N_pad-N are unused.
/// \return 0 on failure
///         1 on success
int generator_RREF(generator_mat_t *G,
                   uint8_t is_pivot_column[N_pad]) {
    uint8_t tmp, sc;

    vec256_t *gm[K] __attribute__((aligned(32)));
    vec256_t em[0x80][NW];
    vec256_t *ep[0x80];
    vec256_t c7f;
    vec256_t x, t, *rp, *rg;

    vset8(c7f, 0x7f);

    for (uint64_t i = 0; i < K; i++) {
        gm[i] = (vec256_t *) G->values[i];
    }

    for (uint64_t i = 0; i < K; i++) {
        uint64_t j = i;
        /*start by searching the pivot in the col = row*/
        uint64_t pivc = i;

        while (pivc < N) {
            while (j < K) {
                sc = G->values[j][pivc];
                if (sc != 0) {
                    goto found;
                }

                j++;
            }
            pivc++;     /* move to next col */
            j = i;      /*starting from row to red */
        }

        if (pivc >= N) {
            /* no pivot candidates left, report failure */
            return 0;
        }

        found:
        is_pivot_column[pivc] = 1; /* pivot found, mark the column*/

        /* if we found the pivot on a row which has an index > pivot_column
         * we need to swap the rows */
        if (i != j) {
            for (uint32_t k = 0; k < NW; k++) {
                t = gm[i][k];
                gm[i][k] = gm[j][k];
                gm[j][k] = t;
            }
        }

        /* Compute rescaling factor */
        /* rescale pivot row to have pivot = 1. Values at the left of the pivot
         * are already set to zero by previous iterations */

        //	generate the em matrix
        rg = gm[i];
        memcpy(em[1], rg, LESS_WSZ * NW);

        for (j = 2; j < 127; j++) {
            for (uint32_t k = 0; k < NW; k++) {
                vadd8(x, em[j - 1][k], rg[k])
                vred8(x, t, c7f);
                em[j][k] = x;
            }
        }

        //	shuffle the pointers into ep
        sc = ((uint8_t *) rg)[pivc];
        ep[0] = em[0];
        tmp = sc;
        for (j = 1; j < 127; j++) {
            ep[tmp] = em[j];
            tmp += sc;
            tmp = (tmp + (tmp >> 7)) & 0x7F;
        }
        ep[0x7F] = em[0];

        //	copy back the normalized one
        memcpy(rg, ep[1], LESS_WSZ * NW);

        /* Subtract the now placed and reduced pivot rows, from the others,
         * after rescaling it */
        for (j = 0; j < K; j++) {
            sc = ((uint8_t *) gm[j])[pivc] & 0x7F;
            if (j != i && (sc != 0x00)) {
                rp = ep[127 - sc];
                for (uint32_t k = 0; k < NW; k++) {
                    vadd8(x, gm[j][k], rp[k])
                    vred8(x, t, c7f);
                    gm[j][k] = x;
                }
            }
        }
    }

    return 1;
} /* end generator_RREF */

/// \param G[in/out]: generator matrix K \times N
/// \param is_pivot_column[out]: N bytes, set to 1 if this column
///                 is a pivot column
/// \param was_pivot_column[out]: N bytes, set to 1 if this column
///                 is a pivot column
/// \param pvt_reuse_limit:[in]:
/// \return 0 on failure
///         1 on success
int generator_RREF_pivot_reuse(generator_mat_t *G,
                               uint8_t is_pivot_column[N],
                               uint8_t was_pivot_column[N],
                               const int pvt_reuse_limit) {
    uint8_t tmp, sc;

    vec256_t *gm[K] __attribute__((aligned(32)));
    vec256_t em[0x80][NW];
    vec256_t *ep[0x80];
    vec256_t c7f;
    vec256_t x, t, *rp, *rg;

    vset8(c7f, 0x7f);
    int pvt_reuse_cnt = 0;

    // this loop roughly takes 2.2% of the whole function runtime ()
    if (pvt_reuse_limit != 0) {
        for (int preproc_col = K - 1; preproc_col >= 0; preproc_col--) {
            if (was_pivot_column[preproc_col] == 1) {
                // find pivot row
                uint32_t pivot_el_row = 0;
                for (uint32_t row = 0; row < K; row = row + 1) {
                    if (G->values[row][preproc_col] != 0) {
                        pivot_el_row = row;
                    }
                }

                swap_rows(G->values[preproc_col], G->values[pivot_el_row]);
            }
        }
    }

    for (uint64_t i = 0; i < K; i++) {
        gm[i] = (vec256_t *) G->values[i];
    }

    for (uint64_t i = 0; i < K; i++) {
        uint64_t j = i;
        /*start by searching the pivot in the col = row*/
        uint64_t pivc = i;

        while (pivc < N) {
            while (j < K) {
                sc = G->values[j][pivc];
                if (sc != 0) {
                    goto found;
                }

                j++;
            }
            pivc++;     /* move to next col */
            j = i;      /*starting from row to red */
        }

        if (pivc >= N) {
            return 0; /* no pivot candidates left, report failure */
        }

        found:
        is_pivot_column[pivc] = 1; /* pivot found, mark the column*/

        /* if we found the pivot on a row which has an index > pivot_column
         * we need to swap the rows */
        if (i != j) {
            was_pivot_column[j] = 0; // pivot no longer reusable - will be corrupted during reduce row
            for (uint32_t k = 0; k < NW; k++) {
                t = gm[i][k];
                gm[i][k] = gm[j][k];
                gm[j][k] = t;
            }
        }

        /// NOTE: this needs explenation. We can skip the reduction of the pivot row, because for
        /// the CF it doesnt matter. The only thing that is important for the CF is the number of
        /// zeros, and this doest change if we reduce a reused pivot row.
        if ((was_pivot_column[pivc] == 1) &&
            (pvt_reuse_cnt < pvt_reuse_limit) &&
            (pivc < K)) {
            pvt_reuse_cnt += 1;
            continue;
        }

        /* Compute rescaling factor */
        /* rescale pivot row to have pivot = 1. Values at the left of the pivot
         * are already set to zero by previous iterations */

        //  generate the em matrix
        rg = gm[i];
        memcpy(em[1], rg, LESS_WSZ * NW);

        for (j = 2; j < 127; j++) {
            LOOP_UNROLL_4
            for (uint32_t k = 0; k < NW; k++) {
                vadd8(x, em[j - 1][k], rg[k])
                const __m256i xx = _mm256_sub_epi8(x, c7f);
#if defined(LESS_USE_BLEND_IN_ARITH)
                const __m256i tt = _mm256_blendv_epi8(xx, x, xx);
#else
                const __m256i tt = _mm256_min_epu8(xx, x);
#endif
                em[j][k] = tt;
            }
        }

        //  shuffle the pointers into ep
        sc = ((uint8_t *) rg)[pivc];
        ep[0] = em[0];
        tmp = sc;
        for (j = 1; j < 127; j++) {
            ep[tmp] = em[j];
            tmp += sc;
            tmp = (tmp + (tmp >> 7)) & 0x7F;
        }
        ep[0x7F] = em[0];

        //  copy back the normalized one
        memcpy(rg, ep[1], LESS_WSZ * NW);

        /* Subtract the now placed and reduced pivot rows, from the others,
         * after rescaling it */
        for (j = 0; j < K; j++) {
            sc = ((uint8_t *) gm[j])[pivc];
            if (sc != 0x00 && j != i) {
                rp = ep[127 - sc];
                LOOP_UNROLL_4
                for (uint32_t k = 0; k < NW; k++) {
                    vadd8(x, gm[j][k], rp[k]);
                    const __m256i xx = _mm256_sub_epi8(x, c7f);
#if defined(LESS_USE_BLEND_IN_ARITH)
                    const __m256i tt = _mm256_blendv_epi8(xx, x, xx);
#else
                    const __m256i tt = _mm256_min_epu8(xx, x);
#endif
                    gm[j][k] = tt;
                }
            }
        }
    }

    return 1;
} /* end generator_RREF_pivot_reuse */

/// NOTE: for AVX2 the code is the same as for the non ct
/// \param G[in/out]: generator matrix K \times N
/// \param is_pivot_column[out]: N bytes, set to 1 if this column
///                 is a pivot column
/// \param was_pivot_column[out]: N bytes, set to 1 if this column
///                 is a pivot column
/// \param pvt_reuse_limit:[in]: number of columns which are at most
///     allowed to be reused.
/// \return 0 on failure
///         1 on success
int generator_RREF_pivot_reuse_ct(generator_mat_t *G,
                               uint8_t is_pivot_column[N],
                               uint8_t was_pivot_column[N],
                               const int pvt_reuse_limit) {
    return generator_RREF_pivot_reuse(G, is_pivot_column, was_pivot_column, pvt_reuse_limit);
} /* end generator_RREF_pivot_reuse_ct */

/// Compresses a generator matrix in RREF into a array of bytes
/// \param compressed[out] byte array of length RREF_MAT_PACKEDBYTES
/// \param full[in]: full generator matrix (K \times N)
/// \param is_pivot_column[in]: array of length N in which K fields are 1, the
///     rest must be zero. Indicating the positions of the pivot columns.
void compress_rref(uint8_t *compressed,
                   const generator_mat_t *const full,
                   const uint8_t is_pivot_column[N]) {
    // Compress pivot flags
    for (uint32_t col_byte = 0; col_byte < N / 8; col_byte++) {
        compressed[col_byte] = is_pivot_column[8 * col_byte + 0] |
                               (is_pivot_column[8 * col_byte + 1] << 1) |
                               (is_pivot_column[8 * col_byte + 2] << 2) |
                               (is_pivot_column[8 * col_byte + 3] << 3) |
                               (is_pivot_column[8 * col_byte + 4] << 4) |
                               (is_pivot_column[8 * col_byte + 5] << 5) |
                               (is_pivot_column[8 * col_byte + 6] << 6) |
                               (is_pivot_column[8 * col_byte + 7] << 7);
    }

#if (CATEGORY == 252) || (CATEGORY == 548)
    // Compress last flags
    compressed[N / 8] = is_pivot_column[N - 4] | (is_pivot_column[N - 3] << 1) |
                        (is_pivot_column[N - 2] << 2) |
                        (is_pivot_column[N - 1] << 3);

    int compress_idx = N / 8 + 1;
#else
    int compress_idx = N / 8;
#endif

    // Compress non-pivot columns row-by-row
    uint8_t encode_state = 0;
    for (uint32_t row_idx = 0; row_idx < K; row_idx++) {
        for (uint32_t col_idx = 0; col_idx < N; col_idx++) {
            if (!is_pivot_column[col_idx]) {
                switch (encode_state) {
                    // NOTE: the default label is only here for the linter.
                    default:
                    case 0:
                        compressed[compress_idx] = full->values[row_idx][col_idx];
                        break;
                    case 1:
                        compressed[compress_idx] =
                                compressed[compress_idx] | (full->values[row_idx][col_idx] << 7);
                        compress_idx++;
                        compressed[compress_idx] = (full->values[row_idx][col_idx] >> 1);
                        break;
                    case 2:
                        compressed[compress_idx] =
                                compressed[compress_idx] | (full->values[row_idx][col_idx] << 6);
                        compress_idx++;
                        compressed[compress_idx] = (full->values[row_idx][col_idx] >> 2);
                        break;
                    case 3:
                        compressed[compress_idx] =
                                compressed[compress_idx] | (full->values[row_idx][col_idx] << 5);
                        compress_idx++;
                        compressed[compress_idx] = (full->values[row_idx][col_idx] >> 3);
                        break;
                    case 4:
                        compressed[compress_idx] =
                                compressed[compress_idx] | (full->values[row_idx][col_idx] << 4);
                        compress_idx++;
                        compressed[compress_idx] = (full->values[row_idx][col_idx] >> 4);
                        break;
                    case 5:
                        compressed[compress_idx] =
                                compressed[compress_idx] | (full->values[row_idx][col_idx] << 3);
                        compress_idx++;
                        compressed[compress_idx] = (full->values[row_idx][col_idx] >> 5);
                        break;
                    case 6:
                        compressed[compress_idx] =
                                compressed[compress_idx] | (full->values[row_idx][col_idx] << 2);
                        compress_idx++;
                        compressed[compress_idx] = (full->values[row_idx][col_idx] >> 6);
                        break;
                    case 7:
                        compressed[compress_idx] =
                                compressed[compress_idx] | (full->values[row_idx][col_idx] << 1);
                        compress_idx++;
                        break;
                }

                if (encode_state != 7) {
                    encode_state++;
                } else {
                    encode_state = 0;
                }
            }
        }
    }
} /* end compress_rref */

/// Expands a compressed RREF generator matrix into a full one
/// \param full[out]: output full matrix (K \times N)
/// \param compressed[in]: bytestream containing the compressed maitrx
/// \param is_pivot_column[out]: N bytes will be initialized with zeros. And
///     only 1 will be written at the column position which is a pivot column.
void expand_to_rref(generator_mat_t *full,
                    const uint8_t *compressed,
                    uint8_t is_pivot_column[N]) {
    // Decompress pivot flags
    for (uint64_t i = 0; i < N; i++) {
        is_pivot_column[i] = 0;
    }

    for (uint64_t col_byte = 0; col_byte < N / 8; col_byte++) {
        is_pivot_column[col_byte * 8 + 0] = compressed[col_byte] & 0x1;
        is_pivot_column[col_byte * 8 + 1] = (compressed[col_byte] >> 1) & 0x1;
        is_pivot_column[col_byte * 8 + 2] = (compressed[col_byte] >> 2) & 0x1;
        is_pivot_column[col_byte * 8 + 3] = (compressed[col_byte] >> 3) & 0x1;
        is_pivot_column[col_byte * 8 + 4] = (compressed[col_byte] >> 4) & 0x1;
        is_pivot_column[col_byte * 8 + 5] = (compressed[col_byte] >> 5) & 0x1;
        is_pivot_column[col_byte * 8 + 6] = (compressed[col_byte] >> 6) & 0x1;
        is_pivot_column[col_byte * 8 + 7] = (compressed[col_byte] >> 7) & 0x1;
    }

#if (CATEGORY == 252) || (CATEGORY == 548)
    // Decompress last flags
    is_pivot_column[N - 4] = compressed[N / 8] & 0x1;
    is_pivot_column[N - 3] = (compressed[N / 8] >> 1) & 0x1;
    is_pivot_column[N - 2] = (compressed[N / 8] >> 2) & 0x1;
    is_pivot_column[N - 1] = (compressed[N / 8] >> 3) & 0x1;

    int compress_idx = N / 8 + 1;
#else
    int compress_idx = N / 8;
#endif

    // Decompress columns row-by-row
    uint8_t decode_state = 0;
    for (uint32_t row_idx = 0; row_idx < K; row_idx++) {
        int pivot_idx = 0;
        for (uint32_t col_idx = 0; col_idx < N; col_idx++) {
            if (!is_pivot_column[col_idx]) {
                // Decompress non-pivot
                switch (decode_state) {
                    // NOTE: the default label is only here for the linter.
                    default:
                    case 0:
                        full->values[row_idx][col_idx] = compressed[compress_idx] & MASK_Q;
                        break;
                    case 1:
                        full->values[row_idx][col_idx] =
                                ((compressed[compress_idx] >> 7) |
                                 (compressed[compress_idx + 1] << 1)) &
                                MASK_Q;
                        compress_idx++;
                        break;
                    case 2:
                        full->values[row_idx][col_idx] =
                                ((compressed[compress_idx] >> 6) |
                                 (compressed[compress_idx + 1] << 2)) &
                                MASK_Q;
                        compress_idx++;
                        break;
                    case 3:
                        full->values[row_idx][col_idx] =
                                ((compressed[compress_idx] >> 5) |
                                 (compressed[compress_idx + 1] << 3)) &
                                MASK_Q;
                        compress_idx++;
                        break;
                    case 4:
                        full->values[row_idx][col_idx] =
                                ((compressed[compress_idx] >> 4) |
                                 (compressed[compress_idx + 1] << 4)) &
                                MASK_Q;
                        compress_idx++;
                        break;
                    case 5:
                        full->values[row_idx][col_idx] =
                                ((compressed[compress_idx] >> 3) |
                                 (compressed[compress_idx + 1] << 5)) &
                                MASK_Q;
                        compress_idx++;
                        break;
                    case 6:
                        full->values[row_idx][col_idx] =
                                ((compressed[compress_idx] >> 2) |
                                 (compressed[compress_idx + 1] << 6)) &
                                MASK_Q;
                        compress_idx++;
                        break;
                    case 7:
                        full->values[row_idx][col_idx] =
                                (compressed[compress_idx] >> 1) & MASK_Q;
                        compress_idx++;
                        break;
                }

                if (decode_state != 7) {
                    decode_state++;
                } else {
                    decode_state = 0;
                }
            } else {
                // Decompress pivot
                full->values[row_idx][col_idx] = ((uint32_t)row_idx == (uint32_t)pivot_idx);
                pivot_idx++;
            }
        }
    }
} /* end expand_to_rref */

/// Expands a compressed RREF generator matrix into a full one
/// \param full[out]: output generator matrix (K \times N) 
/// \param compact[out]: input compressed generator matrix (K \times N-K) 
void generator_rref_expand(generator_mat_t *full, const rref_generator_mat_t *const compact) {
    uint32_t placed_dense_cols = 0;
    for (uint32_t col_idx = 0; col_idx < N; col_idx++) {
        if ((placed_dense_cols < N - K) && (col_idx == compact->column_pos[placed_dense_cols])) {
            /* non-pivot column, restore one full column */
            for (uint32_t row_idx = 0; row_idx < K; row_idx++) {
                full->values[row_idx][col_idx] = compact->values[row_idx][placed_dense_cols];
            }
            placed_dense_cols++;
        } else {
            /* regenerate the appropriate pivot column */
            for (uint32_t row_idx = 0; row_idx < K; row_idx++) {
                full->values[row_idx][col_idx] = (row_idx == col_idx - placed_dense_cols);
            }
        }
    }
} /* end generator_rref_expand */

/// expands a systematic form generator from a seed randomly drawing only
/// non-identity portion
/// \param res[out]: full rank generator matrix K \times N-K
/// \param seed[int] seed for the prng
void generator_sample(rref_generator_mat_t *res, const unsigned char seed[SEED_LENGTH_BYTES]) {
    SHAKE_STATE_STRUCT csprng_state;
    initialize_csprng(&csprng_state, seed, SEED_LENGTH_BYTES);
    sample_antiorthogonal(res->values,seed);
    for (uint32_t i = 0; i < N - K; i++) {
        res->column_pos[i] = i + K;
    }
} /* end generator_sample */

/// V1 =V2
/// \param V1[out]: pointer to generator matrix (non IS part)
/// \param V2[in]: pointer to generator matrix (non IS part)
void normalized_copy(normalized_IS_t *V1,
                     const normalized_IS_t *V2) {
    memcpy(V1->values, V2->values, sizeof(normalized_IS_t));
} /* end normalized_copy */

/// \param V[in/out]: K \times N-K matrix in which row `row1` and
///     row `row2` are swapped
/// \param row1[in]: first row
/// \param row2[in]: second row
void normalized_row_swap(normalized_IS_t *V,
              const POSITION_T row1,
              const POSITION_T row2) {
    for(uint32_t i = 0; i < N-K; i++){
        const POSITION_T tmp = V->values[row1][i];
        V->values[row1][i] = V->values[row2][i];
        V->values[row2][i] = tmp;
    }
} /* normalized_row_swap */

/// right-multiplies a generator by a monomial: res = G*monom
/// \param res[out]: pointer to an uninitialized generator matrix (non IS part)
/// \param G[in]: pointer to an initialized generator matrix (non IS part)
/// \param monom[in]: pointer to an initialized monomial matrix
void normalized_monomial_right(normalized_IS_t *res,
                               const normalized_IS_t *const G,
                               const monomial_t *const monom) {
    FQ_ELEM tmp1[K_pad*K_pad] __attribute__((aligned(32)));
    FQ_ELEM tmp2[K_pad*K_pad] __attribute__((aligned(32)));
    matrix_transpose(tmp1, (uint8_t *)G->values, K_pad, K_pad, K_pad, K_pad);

    for (uint64_t i = 0; i < K; i++) {
        const __m256i p = _mm256_set1_epi16(monom->coefficients[i]);
        const uint64_t in_off = i * K_pad;
        const uint64_t out_off = monom->permutation[i] * K_pad;
        for (uint64_t j = 0; j+32 <= K_pad; j += 32) {
            const __m128i a = _mm_loadu_si128((const __m128i *)(tmp1 + in_off + j +  0));
            const __m128i b = _mm_loadu_si128((const __m128i *)(tmp1 + in_off + j + 16));
            const __m256i t0 = _mm256_cvtepu8_epi16(a);
            const __m256i t1 = _mm256_cvtepu8_epi16(b);

            const __m256i t = avx_mul_full256(t0, t1, p, p);
            _mm256_store_si256((__m256i *)(tmp2 + out_off + j), t);
        }
    }

    matrix_transpose((uint8_t *)res->values, tmp2, K_pad, K_pad, K_pad, K_pad);
} /* normalized_monomial_right*/

/// \param A[out]: pointer to allocated normalized struct, which get filled with the
///     non-IS of the generator matrix G
/// \param G[in]: generator matrix to extract the non-IS from.
/// \param is_pivot_column[in]: array identifying a pivot column via a 1
void normalized_copy_from_generator_non_information_set(normalized_IS_t *A ,
                                                        const generator_mat_t *const G,
                                                        const uint8_t *const is_pivot_column) {
    // we simply copy the last N-K columns even if they are not the information set.
    for (uint64_t i = 0; i < K; i++) {
        memcpy((uint8_t *)A->values[i], ((uint8_t *)G->values[i]) + K, K);
    }

    // now we scan if we need to fix the non information set
    uint32_t ctr = 0;
    for (; ctr < K && is_pivot_column[ctr] == 1; ctr++) {}

    // easy part: the last N-K columns are the non IS
    if (ctr == K) { return; }

    // "hard" part: copy all remaining columns < K into the non information set part
    for(uint32_t j = 0; j < N-K && is_pivot_column[ctr] == 0; j++) {
        /// copy column
        for (uint32_t k = 0; k < K; k++) {
            A->values[k][j] = G->values[k][ctr];
        }
        ctr += 1;
    }
}


/// \param res[out]: full rank generator matrix K \times K
/// \param seed[int] seed for the prng
void sample_antiorthogonal(FQ_ELEM A[K][K_pad], const unsigned char seed[SEED_LENGTH_BYTES]) {
    SHAKE_STATE_STRUCT csprng_state;
    initialize_csprng(&csprng_state, seed, SEED_LENGTH_BYTES);
    FQ_ELEM G[K_pad][K_pad] __attribute__((aligned(32))) = {0};
    FQ_ELEM c[K_pad] = {0};

    while(c[0] == 0 || !(anti_normalize(c))){
        rand_range_q_state_elements(&csprng_state, c, K);
    }

    set_row(A[0],c);
    row_mul(c,fq_inv(c[0]));
    set_row(G[0],c);

    FQ_ELEM M[K_pad][K_pad] __attribute__((aligned(32))) = {0};
    uint16_t kp = 1;

    int iter = 0;
    int max_iter = 1000000;


    while (kp < K) {

        for(int i=0; i<kp; i++){
            memset(M[i],0,K_pad);
            for(int j=0; j<K-kp; j++){
                M[i][j] = G[i][j+kp];
            }
        }

        if(iter < max_iter){

            FQ_ELEM u[K-kp];

            do{
                rand_range_q_state_elements(&csprng_state, u, K-kp);
                superfast_row_mat_mult(c,u,M,K-kp,kp);

                for(int i=0; i<K-kp; i++){
                    c[kp + i] = fq_opp(u[i]);
                }


                uint16_t inner_prod_of_sampled = 0;

                for(int j = 0; j<K; j++){
                    inner_prod_of_sampled = (inner_prod_of_sampled + c[j]*G[0][j]) % 127;
                }


                iter += 1;

            }while(!anti_normalize(c) & (iter < max_iter));

            if(iter >= max_iter){
                break;
            }

            set_row(A[kp],c);
            set_row(G[kp],c);
        
            // Set 0 all elements of last row up to kp-1 
            for(uint16_t i=0; i<kp; i++){
                if(G[kp][i] != 0){
                    row_sum(G[kp],G[i],fq_opp(G[kp][i]),K);
                }
            }

            if(G[kp][kp] == 0){
                for(uint16_t i=kp+1; i<K; i++){
                    if(G[kp][i] != 0){
                        swap_columns(G,kp,i,kp+1);
                        swap_columns(A,kp,i,kp+1);
                        break;
                    }
                }
            }
            
            row_mul(G[kp],fq_inv(G[kp][kp]));

            // Set all elements in that column to 0 with row sum
            // Optimization: subdivide in blocks of 32 bytes (AVX2 registers size)
            // Sum rows starting from the 32-block in which kp is contained
            // Everything before that is already set to 0, no need to sum
            uint16_t sum_start = kp/32;
            for(int i=0; i<kp; i++){
                if(G[i][kp] != 0){
                    row_sum(G[i]+sum_start*32, G[kp]+sum_start*32, fq_opp(G[i][kp]), K-(sum_start*32));
                }
            }

            kp += 1;
            iter += 1;

        } else{

            for(uint16_t i = 0; i<K_pad; i++){
                memset(G[i],0,K_pad);

            }
            memset(c,0,K_pad);

            c[0] = 0;
            while(c[0] == 0 || !(anti_normalize(c))){
                rand_range_q_state_elements(&csprng_state, c, K);
            }

            set_row(A[0],c);
            row_mul(c,fq_inv(c[0]));
            set_row(G[0],c);

            kp = 1;
            iter = 0;
        }
    }
}


void swap_columns(FQ_ELEM M[K][K_pad], uint16_t c1, uint16_t c2, uint16_t r){
    FQ_ELEM tmp;
    for(uint16_t i=0; i<r; i++){
        tmp = M[i][c1];
        M[i][c1] = M[i][c2];
        M[i][c2] = tmp;
    }
}

void compress_self_orthogonal_noavx(FQ_ELEM PACKED[RREF_AO_BYTES],
        FQ_ELEM A[K][K_pad],
        FQ_ELEM extra_vars[K],
        uint64_t bitstring[BITSTRING_LEN]
        ){

    FQ_ELEM M[K][K_pad] = {0};
    memcpy(M,A,K*K_pad); 

    // First bit
    if (A[0][0] > 63){
        bitstring[0] = 1;
    }

    uint8_t pivoted[N-K] = {0}; // 1 => col is pivoted, 0 otherwise
    uint16_t rows[N-K] = {0}; // row on which the column got pivoted 
     
    FQ_ELEM f; // temporary factor variable
    uint16_t x_idx; // index of free variable 
    FQ_ELEM x; // free var value 
    uint16_t extra_num = 0; // number of extra vars
    FQ_ELEM v[K_pad] = {0}; // vector of alpha_d (see paper)
    FQ_ELEM a,b,c; // quadratic equation coefficients
    FQ_ELEM disc; // discriminant value 

    for(uint16_t i=1; i<K; i++){

        // Delete values on new row using pivoted columns
        for(uint16_t col=0; col<i+1; col++){
            if (pivoted[col] == 1){
                f = M[i-1][col];
                for(uint16_t j=0; j<(N-K); j++){
                    M[i-1][j] = fq_sub(M[i-1][j],fq_mul(M[rows[col]][j],f));
                }
            }
        }

        // Try to pivot an unpivoted column on the new row
        for(uint16_t col=0; col<i+1; col++){
            if (pivoted[col] == 0 && M[i-1][col] != 0) {
            
                // Normalize row
                f = fq_inv(M[i-1][col]);
                for (uint16_t j=0; j<(N-K);j++){
                    // If the column is already pivoted the value is 0
                    if (pivoted[j] == 0)
                        M[i-1][j] = fq_mul(M[i-1][j],f);
                }

                // Delete other rows
                for (uint16_t j=0; j<(i-1);j++){
                    f = M[j][col];
                    for (uint16_t l=0; l<(N-K);l++){
                        M[j][l] = fq_sub(M[j][l],fq_mul(M[i-1][l],f));
                    }
                }

                pivoted[col] = 1;
                rows[col] = i-1;
                break;
            }
        }

        
        // Determine free variable and communicate extra_vars 
        x_idx = UINT16_MAX;;
        for (uint16_t j=0; j<(i+1);j++){
            if(pivoted[j] == 0  && x_idx == UINT16_MAX){
                x_idx = j;
            }else if (pivoted[j] == 0){
                extra_vars[extra_num] = A[i][j];
                extra_num +=1;
            }
        }

        // Compute all alpha_d
        memset(v,0,i);

        for (uint16_t j=0; j<(i+1);j++){
            if(pivoted[j] == 1){
                for (uint16_t l=0; l<(i+1); l++){
                    if(pivoted[l] == 0 && l != x_idx){
                        v[j] = fq_sub(v[j],fq_mul(M[rows[j]][l],M[i][l]));
                    }
                }

                for (uint16_t l=(i+1); l<(N-K);l++){
                        v[j] = fq_sub(v[j],fq_mul(M[rows[j]][l],M[i][l]));
                }

            }
        }

        // Set and solve the quadratic equation 

        a = 1;
        b = 0;
        c = 1;

        for (uint16_t j=0; j<(i+1);j++){
            if(pivoted[j] == 1){
                a = fq_add(a,fq_square(M[rows[j]][x_idx]));
                b = fq_sub(b,fq_mul(M[rows[j]][x_idx],v[j]));
                c = fq_add(c,fq_square(v[j]));
            } else if (x_idx != j) {
                c = fq_add(c,fq_square(M[i][j]));
            }
        }

        for (uint16_t j=i+1; j<(N-K);j++){
            c = fq_add(c,fq_square(M[i][j]));
        }

        b = fq_mul(b,2);

        if (a == 0){
            x = fq_opp(fq_mul(c,fq_inv(b)));
            if(b==0){ 
                extra_vars[extra_num] = A[i][x_idx];
                extra_num += 1;
            }
        } else {
            disc = fq_sub(fq_square(b),fq_mul(4,fq_mul(a,c)));
            x = fq_mul(fq_inv(fq_mul(2,a)),fq_add(fq_opp(b),fq_sqrt(disc)));

            if (x != A[i][x_idx]){
                bitstring[i>>6] += (UINT64_C(1)<<(i % 64));
            }
        }

        memcpy(M[i],A[i],i+1); 
    }

    uint32_t idx = 0;
    for(uint16_t i=0; i<K; i++){
        for(uint16_t j=(i+1); j<K; j++){
            PACKED[idx] = A[i][j];
            idx+=1;
        }
    }
}

void recover_self_orthogonal_noavx(FQ_ELEM A[K][K_pad],
        FQ_ELEM PACKED[RREF_AO_BYTES],
        FQ_ELEM extra_vars[K],
        uint64_t bitstring[BITSTRING_LEN]
        ){

    FQ_ELEM M[K][K_pad] = {0};

    uint32_t idx = 0;
    for(uint16_t i=0; i<K; i++){
        for(uint16_t j=(i+1); j<K; j++){
            A[i][j] = PACKED[idx];
            M[i][j] = PACKED[idx];
            idx+=1;
        }
    }

    uint8_t pivoted[N-K] = {0}; // 1 => col is pivoted, 0 otherwise
    uint16_t rows[N-K] = {0}; // row on which the column got pivoted 
     
    FQ_ELEM f; // temporary factor variable
    uint16_t x_idx; // index of free variable 
    FQ_ELEM x; // free var value 
    uint16_t extra_num = 0; // number of extra vars
    FQ_ELEM v[K_pad] = {0}; // vector of alpha_d (see paper)
    FQ_ELEM a,b,c; // quadratic equation coefficients
    FQ_ELEM disc; // discriminant value 
                  
    uint16_t d = 1;
    for (uint16_t i = 1; i<(K); i++){
         d += fq_square(M[0][i]);
    }
    d = fq_red(d);
    d = fq_opp(d);
    
    if((bitstring[0] & 1) == 1){
        A[0][0] = fq_opp(fq_sqrt(d));
        M[0][0] = A[0][0]; 
    } else {
        A[0][0] = fq_sqrt(d);
        M[0][0] = A[0][0];
    }

    for(uint16_t i=1; i<K; i++){

        // Delete values on new row using pivoted columns
        for(uint16_t col=0; col<i+1; col++){
            if (pivoted[col] == 1){
                f = M[i-1][col];
                for(uint16_t j=0; j<(N-K); j++){
                    M[i-1][j] = fq_sub(M[i-1][j],fq_mul(M[rows[col]][j],f));
                }
            }
        }

        // Try to pivot an unpivoted column on the new row
        for(uint16_t col=0; col<i+1; col++){
            if (pivoted[col] == 0 && M[i-1][col] != 0) {
            
                // Normalize row
                f = fq_inv(M[i-1][col]);
                for (uint16_t j=0; j<(N-K);j++){
                    // If the column is already pivoted the value is 0
                    if (pivoted[j] == 0)
                        M[i-1][j] = fq_mul(M[i-1][j],f);
                }

                // Delete other rows
                for (uint16_t j=0; j<(i-1);j++){
                    f = M[j][col];
                    for (uint16_t l=0; l<(N-K);l++){
                            M[j][l] = fq_sub(M[j][l],fq_mul(M[i-1][l],f));
                    }
                }

                pivoted[col] = 1;
                rows[col] = i-1;
                break;
            }
        }

        
        // Determine free variable and communicate extra_vars 
        x_idx = UINT16_MAX;;
        for (uint16_t j=0; j<(i+1);j++){
            if(pivoted[j] == 0  && x_idx == UINT16_MAX){
                x_idx = j;
            }else if (pivoted[j] == 0){
                A[i][j] = extra_vars[extra_num];
                M[i][j] = extra_vars[extra_num];
                extra_num += 1;
            }
        }

        // Compute all alpha_d
        memset(v,0,i);

        for (uint16_t j=0; j<(i+1);j++){
            if(pivoted[j] == 1){
                for (uint16_t l=0; l<(i+1); l++){
                    if(pivoted[l] == 0 && l != x_idx){
                        v[j] = fq_opp(fq_mul(M[rows[j]][l],M[i][l]));
                    }
                }

                for (uint16_t l=(i+1); l<(N-K);l++){
                        v[j] = fq_sub(v[j],fq_mul(M[rows[j]][l],M[i][l]));
                }

            }
        }

        // Set and solve the quadratic equation 
        
        a = 1;
        b = 0;
        c = 1;

        for (uint16_t j=0; j<(i+1);j++){
            if(pivoted[j] == 1){
                a = fq_add(a,fq_square(M[rows[j]][x_idx]));
                b = fq_sub(b,fq_mul(M[rows[j]][x_idx],v[j]));
                c = fq_add(c,fq_square(v[j]));
            } else if (x_idx != j) {
                c = fq_add(c,fq_square(M[i][j]));
            }
        }

        for (uint16_t j=i+1; j<(N-K);j++){
            c = fq_add(c,fq_square(M[i][j]));
        }

        b = fq_mul(b,2);

        if (a == 0){
            if(b ==0){ 
                M[i][x_idx] = extra_vars[extra_num];
                A[i][x_idx] = extra_vars[extra_num];
                extra_num +=1;
            }else{
                x = fq_opp(fq_mul(c,fq_inv(b)));
                M[i][x_idx] = x;
                A[i][x_idx] = x;
            }
        } else {
            disc = fq_sub(fq_square(b),fq_mul(4,fq_mul(a,c)));

            if ((bitstring[i>>6] & (UINT64_C(1)<<i)) == (UINT64_C(1)<<i)){
                x = fq_mul(fq_inv(fq_mul(2,a)),fq_sub(fq_opp(b),fq_sqrt(disc)));
                M[i][x_idx] = x;
                A[i][x_idx] = x;
            } else {
                x = fq_mul(fq_inv(fq_mul(2,a)),fq_add(fq_opp(b),fq_sqrt(disc)));
                M[i][x_idx] = x;
                A[i][x_idx] = x;
            }

        }

        // Recompute variables back
        for (uint16_t j = 0; j<(i+1); j++){
            if(pivoted[j] == 1){
                M[i][j] = fq_add(fq_opp(fq_mul(M[rows[j]][x_idx],M[i][x_idx])),v[j]);
                A[i][j] = M[i][j];
            }
        }
    }
}


void compress_self_orthogonal(FQ_ELEM PACKED[RREF_AO_BYTES],
        FQ_ELEM A[K][K_pad],
        FQ_ELEM extra_vars[K],
        uint64_t bitstring[BITSTRING_LEN]
        ){

    FQ_ELEM M[K][K_pad] = {0};
    memcpy(M,A,K*K_pad); 

    // First bit
    if (A[0][0] > 63){
        bitstring[0] = 1;
    }

    uint8_t pivoted[N-K] = {0}; // 1 => col is pivoted, 0 otherwise
    uint16_t rows[N-K] = {0}; // row on which the column got pivoted 
     
    FQ_ELEM f; // temporary factor variable
    uint16_t x_idx; // index of free variable 
    FQ_ELEM x; // free var value 
    uint16_t extra_num = 0; // number of extra vars
    FQ_ELEM v[K_pad] = {0}; // vector of alpha_d (see paper)
    FQ_ELEM a,b,c; // quadratic equation coefficients
    FQ_ELEM disc; // discriminant value 

    for(uint16_t i=1; i<K; i++){

        // Delete values on new row using pivoted columns
        for(uint16_t col=0; col<i+1; col++){
            if (pivoted[col] == 1){
                f = M[i-1][col];
                row_sum(M[i-1],M[rows[col]],fq_opp(f),K);
            }
        }

        // Try to pivot an unpivoted column on the new row
        for(uint16_t col=0; col<i+1; col++){
            if (pivoted[col] == 0 && M[i-1][col] != 0) {
            
                // Normalize row
                row_mul(M[i-1],fq_inv(M[i-1][col]));

                // Delete other rows
                for (uint16_t j=0; j<(i-1);j++){
                    row_sum(M[j], M[i-1], fq_opp(M[j][col]), N-K);
                    M[j][col] = 0; 
                }

                pivoted[col] = 1;
                rows[col] = i-1;
                break;
            }
        }

        // Compute all alpha_d
        memset(v,0,i+1);
        
        // Determine free variable, communicate extra_vars and compute v
        x_idx = UINT16_MAX;;
        for (uint16_t j=0; j<(i+1);j++){
            if(pivoted[j] == 0  && x_idx == UINT16_MAX){
                x_idx = j;
            }else if (pivoted[j] == 0){

                extra_vars[extra_num] = A[i][j];
                M[i][j] = A[i][j];
                extra_num += 1;

                for (uint16_t l=0; l<(i+1);l++){
                    if(pivoted[l] == 1){
                        v[l] = fq_sub(v[l],fq_mul(M[rows[l]][j],M[i][j]));
                    }
                }

            } else if (pivoted[j] == 1) {

                uint16_t parallel_cols = ((K_pad-i-1) >> 5) << 5;
                if(parallel_cols != 0) v[j]  =  fq_sub(v[j],superfast_scalar_prod(M[rows[j]]+(i+1),A[i]+(i+1),parallel_cols));
                if (parallel_cols + i + 1 < K){
                    for (uint16_t l=(parallel_cols + i + 1); l<K;l++){
                        v[j] = fq_sub(v[j],fq_mul(M[rows[j]][l],A[i][l]));
                    }
                }

            }

        }

        // Set and solve the quadratic equation 

        a = 1;
        b = 0;
        c = 1;

        for (uint16_t j=0; j<(i+1);j++){
            if(pivoted[j] == 1){
                a = fq_add(a,fq_square(M[rows[j]][x_idx]));
                b = fq_sub(b,fq_mul(M[rows[j]][x_idx],v[j]));
                c = fq_add(c,fq_square(v[j]));
            } else if (x_idx != j) {
                c = fq_add(c,fq_square(M[i][j]));
            }
        }

        for (uint16_t j=i+1; j<(N-K);j++){
            c = fq_add(c,fq_square(M[i][j]));
        }

        b = fq_add(b,b);

        if (a != 0){
            disc = fq_sub(fq_square(b),fq_mul(4,fq_mul(a,c)));
            x = fq_mul(fq_inv(fq_add(a,a)),fq_add(fq_opp(b),fq_sqrt(disc)));

            if (x != A[i][x_idx]){
                bitstring[i>>6] += (UINT64_C(1)<<(i % 64));
            }

        }else{
            if(b==0){ 
                extra_vars[extra_num] = A[i][x_idx];
                extra_num += 1;
            }
        }

        // Copy missing variables
        memcpy(M[i],A[i],i+1); 
        
    }

    uint32_t idx = 0;
    for(uint16_t i=0; i<K; i++){
        for(uint16_t j=(i+1); j<K; j++){
            PACKED[idx] = A[i][j];
            idx+=1;
        }
    }
}

void recover_self_orthogonal(FQ_ELEM A[K][K_pad],
        FQ_ELEM PACKED[RREF_AO_BYTES],
        FQ_ELEM extra_vars[K],
        uint64_t bitstring[BITSTRING_LEN]
        ){

    FQ_ELEM M[K][K_pad] = {0};

    uint32_t idx = 0;
    for(uint16_t i=0; i<K; i++){
        for(uint16_t j=(i+1); j<K; j++){
            A[i][j] = PACKED[idx];
            M[i][j] = PACKED[idx];
            idx+=1;
        }
    }

    uint8_t pivoted[N-K] = {0}; // 1 => col is pivoted, 0 otherwise
    uint16_t rows[N-K] = {0}; // row on which the column got pivoted 
     
    FQ_ELEM f; // temporary factor variable
    uint16_t x_idx; // index of free variable 
    FQ_ELEM x; // free var value 
    uint16_t extra_num = 0; // number of extra vars
    FQ_ELEM v[K_pad] = {0}; // vector of alpha_d (see paper)
    FQ_ELEM a,b,c; // quadratic equation coefficients
    FQ_ELEM disc; // discriminant value 
                  
    // Recover first position
    //FQ_ELEM d = 126;
    //for (uint16_t i = 1; i<(K); i++){
    //     d = fq_sub(d,fq_square(M[0][i]));
    //}
    // we have -b here at this point
    uint16_t d = 1;
    for (uint16_t i = 1; i<(K); i++){
         d += fq_square(M[0][i]);
         d = fq_red(d);
    }
    d = fq_opp(d);
    
    if((bitstring[0] & 1) == 1){
        A[0][0] = fq_opp(fq_sqrt(d));
        M[0][0] = A[0][0]; 
    } else {
        A[0][0] = fq_sqrt(d);
        M[0][0] = A[0][0];
    }

    for(uint16_t i=1; i<K; i++){

        // Delete values on new row using pivoted columns
        for(uint16_t col=0; col<i+1; col++){
            if (pivoted[col] == 1){
                f = M[i-1][col];
                row_sum(M[i-1],M[rows[col]],fq_opp(f),K);
            }
        }

        // Try to pivot an unpivoted column on the new row
        for(uint16_t col=0; col<i+1; col++){
            if (pivoted[col] == 0 && M[i-1][col] != 0) {
            
                // Normalize row
                row_mul(M[i-1],fq_inv(M[i-1][col]));

                // Delete other rows
                for (uint16_t j=0; j<(i-1);j++){
                    // First right part
                    //row_sum(M[j]+i, M[i-1]+i, fq_opp(M[j][col]), K-i);
                    row_sum(M[j], M[i-1], fq_opp(M[j][col]), K);
                    // Only column
                    // M[j][col] = 0; 

                }

                pivoted[col] = 1;
                rows[col] = i-1;
                break;
            }
        }

        // Compute all alpha_d
        memset(v,0,i+1);
        
        // Determine free variable, communicate extra_vars and compute v
        x_idx = UINT16_MAX;
        for (uint16_t j=0; j<(i+1);j++){
            if(pivoted[j] == 0  && x_idx == UINT16_MAX){
                x_idx = j;
            }else if (pivoted[j] == 0){
                A[i][j] = extra_vars[extra_num];
                M[i][j] = extra_vars[extra_num];
                extra_num += 1;
                for (uint16_t l=0; l<(i+1);l++){
                    if(pivoted[l] == 1){
                        v[l] = fq_sub(v[l],fq_mul(M[rows[l]][j],M[i][j]));
                    }
                }
            } else if (pivoted[j] == 1) {
                uint16_t parallel_cols = ((K_pad-i-1) >> 5) << 5;
                if(parallel_cols != 0) v[j]  =  fq_sub(v[j],superfast_scalar_prod(M[rows[j]]+(i+1),M[i]+(i+1),parallel_cols));
                if (parallel_cols + i + 1 < K){
                    for (uint16_t l=(parallel_cols + i + 1); l<K;l++){
                        v[j] = fq_sub(v[j],fq_mul(M[rows[j]][l],M[i][l]));
                    }
                }
            }

        }

        // Set and solve the quadratic equation 

        a = 1;
        b = 0;
        c = 1;

        for (uint16_t j=0; j<(i+1);j++){
            if(pivoted[j] == 1){
                a = fq_add(a,fq_square(M[rows[j]][x_idx]));
                b = fq_sub(b,fq_mul(M[rows[j]][x_idx],v[j]));
                c = fq_add(c,fq_square(v[j]));
            } else if (x_idx != j) {
                c = fq_add(c,fq_square(M[i][j]));
            }
        }

        for (uint16_t j=i+1; j<(N-K);j++){
            c = fq_add(c,fq_square(M[i][j]));
        }

        b = fq_add(b,b);

        if (a == 0){
            if(b ==0){ 
                M[i][x_idx] = extra_vars[extra_num];
                A[i][x_idx] = extra_vars[extra_num];
                extra_num +=1;
            }else{
                x = fq_opp(fq_mul(c,fq_inv(b)));
                M[i][x_idx] = x;
                A[i][x_idx] = x;
            }
        } else {
            disc = fq_sub(fq_square(b),fq_mul(4,fq_mul(a,c)));

            if ((bitstring[i>>6] & (UINT64_C(1)<<i)) == (UINT64_C(1)<<i)){
                x = fq_mul(fq_inv(fq_add(a,a)),fq_sub(fq_opp(b),fq_sqrt(disc)));
                M[i][x_idx] = x;
                A[i][x_idx] = x;
            } else {
                x = fq_mul(fq_inv(fq_add(a,a)),fq_add(fq_opp(b),fq_sqrt(disc)));
                M[i][x_idx] = x;
                A[i][x_idx] = x;
            }

        }

        // Recompute variables back
        for (uint16_t j = 0; j<(i+1); j++){
            if(pivoted[j] == 1){
                M[i][j] = fq_add(fq_opp(fq_mul(M[rows[j]][x_idx],M[i][x_idx])),v[j]);
                A[i][j] = M[i][j];
            }
        }
    }
}

/* Compresses a generator matrix in RREF storing only non-pivot columns and
 * their position */
void generator_rref_compact(rref_generator_mat_t *compact,
                            const generator_mat_t *const full,
                            const uint8_t is_pivot_column[N]) {
    int dst_col_idx = 0;
    for (uint32_t src_col_idx = 0; src_col_idx < N; src_col_idx++) {
        if (!is_pivot_column[src_col_idx]) {
            for (uint32_t row_idx = 0; row_idx < K; row_idx++) {
                compact->values[row_idx][dst_col_idx] = full->values[row_idx][src_col_idx];
            }
            compact->column_pos[dst_col_idx] = src_col_idx;
            dst_col_idx++;
        }
    }
} /* end generator_rref_compact */

/// Packs one 7-bit field element into the bitstream.
/// Writes with assignment when it opens a new byte, so no memset is needed.
/// \param compressed[out]: destination byte array
/// \param bit_idx[in]: bit position at which to write (advanced by the caller)
/// \param v[in]: value to pack, must be < 128
static inline void pack_fq(uint8_t *compressed,
                           const size_t bit_idx,
                           const FQ_ELEM v) {
    const size_t byte = bit_idx >> 3;
    const uint32_t off = (uint32_t)(bit_idx & 7u);

    if (off == 0) {
        compressed[byte] = (uint8_t)v;          /* opens a new byte */
    } else {
        compressed[byte] |= (uint8_t)(v << off);
    }

    /* off == 0 or 1 means the 7 bits fit entirely in the current byte */
    if (off > 1) {
        compressed[byte + 1] = (uint8_t)(v >> (8u - off));
    }
}

/// Compresses an anti-orthogonal RREF representation into an array of bytes
/// \param compressed[out]: byte array of length RREF_AO_BYTES
/// \param values[in]: flat array of the RREF_AO_LEN non-pivot field elements
/// \param is_pivot_column[in]: array of length N in which K fields are 1, the
///     rest must be zero. Indicating the positions of the pivot columns.
void compress_rref_ao(uint8_t *compressed,
                      const FQ_ELEM *const values,
                      const uint8_t is_pivot_column[N]) {
    // Compress pivot flags
    for (uint32_t col_byte = 0; col_byte < N / 8; col_byte++) {
        compressed[col_byte] = is_pivot_column[8 * col_byte + 0] |
                               (is_pivot_column[8 * col_byte + 1] << 1) |
                               (is_pivot_column[8 * col_byte + 2] << 2) |
                               (is_pivot_column[8 * col_byte + 3] << 3) |
                               (is_pivot_column[8 * col_byte + 4] << 4) |
                               (is_pivot_column[8 * col_byte + 5] << 5) |
                               (is_pivot_column[8 * col_byte + 6] << 6) |
                               (is_pivot_column[8 * col_byte + 7] << 7);
    }

#if (CATEGORY == 252) || (CATEGORY == 548)
    // Compress last flags
    compressed[N / 8] = is_pivot_column[N - 4] | (is_pivot_column[N - 3] << 1) |
                        (is_pivot_column[N - 2] << 2) |
                        (is_pivot_column[N - 1] << 3);

    const size_t compress_idx = N / 8 + 1;
#else
    const size_t compress_idx = N / 8;
#endif

    // Compress the field elements, 7 bits each, starting at compress_idx
    size_t bit_idx = compress_idx * 8u;
    for (uint32_t i = 0; i < RREF_AO_BYTES; i++) {
        pack_fq(compressed, bit_idx, values[i]);
        bit_idx += 7u;
    }
} /* end compress_rref_ao */

void expand_rref_ao(FQ_ELEM *values,
                    const uint8_t *const compressed,
                    uint8_t is_pivot_column[N]) {

    // Decompress pivot flags
    for (uint64_t i = 0; i < N; i++) {
        is_pivot_column[i] = 0;
    }

    for (uint64_t col_byte = 0; col_byte < N / 8; col_byte++) {
        is_pivot_column[col_byte * 8 + 0] = compressed[col_byte] & 0x1;
        is_pivot_column[col_byte * 8 + 1] = (compressed[col_byte] >> 1) & 0x1;
        is_pivot_column[col_byte * 8 + 2] = (compressed[col_byte] >> 2) & 0x1;
        is_pivot_column[col_byte * 8 + 3] = (compressed[col_byte] >> 3) & 0x1;
        is_pivot_column[col_byte * 8 + 4] = (compressed[col_byte] >> 4) & 0x1;
        is_pivot_column[col_byte * 8 + 5] = (compressed[col_byte] >> 5) & 0x1;
        is_pivot_column[col_byte * 8 + 6] = (compressed[col_byte] >> 6) & 0x1;
        is_pivot_column[col_byte * 8 + 7] = (compressed[col_byte] >> 7) & 0x1;
    }

#if (CATEGORY == 252) || (CATEGORY == 548)
    // Decompress last flags
    is_pivot_column[N - 4] = compressed[N / 8] & 0x1;
    is_pivot_column[N - 3] = (compressed[N / 8] >> 1) & 0x1;
    is_pivot_column[N - 2] = (compressed[N / 8] >> 2) & 0x1;
    is_pivot_column[N - 1] = (compressed[N / 8] >> 3) & 0x1;

    int compress_idx = N / 8 + 1;
#else
    int compress_idx = N / 8;
#endif

    #if (CATEGORY == 252) || (CATEGORY == 548)
    size_t bit_idx = (N / 8 + 1) * 8u;
#else
    size_t bit_idx = (N / 8) * 8u;
#endif

    for (uint32_t i = 0; i < RREF_AO_BYTES; i++) {
        const size_t byte = bit_idx >> 3;
        const uint32_t off = (uint32_t)(bit_idx & 7u);
        uint16_t w = compressed[byte] >> off;
        if (off > 1) {
            w |= (uint16_t)compressed[byte + 1] << (8u - off);
        }
        values[i] = (FQ_ELEM)(w & 0x7Fu);
        bit_idx += 7u;
    }
}
