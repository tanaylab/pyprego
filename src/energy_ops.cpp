// energy_ops.cpp -- PSSM energy / derivative computation kernels.
//
// Provides pyprego_init_energies: a C++ implementation of _PWMLRegression.init_energies()
// with OpenMP parallelism over sequences.

#include "_pyprego.h"
#include "dna_utils.h"
#include "log_sum_exp.h"

#include <algorithm>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// pyprego_init_energies
// ---------------------------------------------------------------------------
// Python signature:
//   init_energies(encoded, nuc_factors, spat_factors, train_mask,
//                 spat_bin_size, bidirect, symmetrize_spat,
//                 derivs, spat_derivs) -> None
//
// Parameters:
//   encoded       : ndarray[int8, (N, L)]       - encoded sequences
//   nuc_factors   : ndarray[float64, (K, 4)]    - PSSM probability matrix
//   spat_factors  : ndarray[float64, (B,)]      - spatial factors
//   train_mask    : ndarray[bool, (N,)]          - which sequences to process
//   spat_bin_size : int                          - spatial bin size
//   bidirect      : int                          - 1 if bidirectional, 0 otherwise
//   symmetrize_spat: int                         - 1 if symmetrize spatial bins
//   derivs        : ndarray[float64, (N, K, 4)]  - output: nucleotide derivatives
//   spat_derivs   : ndarray[float64, (N, B)]     - output: spatial derivatives

PyObject *pyprego_init_energies(PyObject * /*self*/, PyObject *args)
{
    PyArrayObject *py_encoded = nullptr;
    PyArrayObject *py_nuc_factors = nullptr;
    PyArrayObject *py_spat_factors = nullptr;
    PyArrayObject *py_train_mask = nullptr;
    int spat_bin_size = 0;
    int bidirect = 0;
    int symmetrize_spat = 0;
    PyArrayObject *py_derivs = nullptr;
    PyArrayObject *py_spat_derivs = nullptr;

    if (!PyArg_ParseTuple(args, "O!O!O!O!iiiO!O!",
                          &PyArray_Type, &py_encoded,
                          &PyArray_Type, &py_nuc_factors,
                          &PyArray_Type, &py_spat_factors,
                          &PyArray_Type, &py_train_mask,
                          &spat_bin_size,
                          &bidirect,
                          &symmetrize_spat,
                          &PyArray_Type, &py_derivs,
                          &PyArray_Type, &py_spat_derivs))
        return NULL;

    // Validate shapes
    if (PyArray_NDIM(py_encoded) != 2) {
        PyErr_SetString(PyExc_ValueError, "encoded must be 2-D (N, L)");
        return NULL;
    }
    if (PyArray_NDIM(py_nuc_factors) != 2) {
        PyErr_SetString(PyExc_ValueError, "nuc_factors must be 2-D (K, 4)");
        return NULL;
    }
    if (PyArray_NDIM(py_spat_factors) != 1) {
        PyErr_SetString(PyExc_ValueError, "spat_factors must be 1-D (B,)");
        return NULL;
    }
    if (PyArray_NDIM(py_train_mask) != 1) {
        PyErr_SetString(PyExc_ValueError, "train_mask must be 1-D (N,)");
        return NULL;
    }
    if (PyArray_NDIM(py_derivs) != 3) {
        PyErr_SetString(PyExc_ValueError, "derivs must be 3-D (N, K, 4)");
        return NULL;
    }
    if (PyArray_NDIM(py_spat_derivs) != 2) {
        PyErr_SetString(PyExc_ValueError, "spat_derivs must be 2-D (N, B)");
        return NULL;
    }

    // Get dimensions
    const npy_intp N = PyArray_DIM(py_encoded, 0);
    const npy_intp L = PyArray_DIM(py_encoded, 1);
    const npy_intp K = PyArray_DIM(py_nuc_factors, 0);
    const npy_intp B = PyArray_DIM(py_spat_factors, 0);

    if (K > L) {
        // num_wins <= 0; just zero the outputs and return
        // Zero derivs and spat_derivs
        memset(PyArray_DATA(py_derivs), 0, (size_t)(N * K * 4) * sizeof(double));
        memset(PyArray_DATA(py_spat_derivs), 0, (size_t)(N * B) * sizeof(double));
        Py_RETURN_NONE;
    }

    const npy_intp num_wins = L - K + 1;

    // Get data pointers
    const int8_t *encoded = (const int8_t *)PyArray_DATA(py_encoded);
    const double *nuc_factors = (const double *)PyArray_DATA(py_nuc_factors);
    const double *spat_factors = (const double *)PyArray_DATA(py_spat_factors);
    const npy_bool *train_mask = (const npy_bool *)PyArray_DATA(py_train_mask);
    double *derivs = (double *)PyArray_DATA(py_derivs);
    double *spat_derivs = (double *)PyArray_DATA(py_spat_derivs);

    // Strides for encoded: (N, L) with int8
    const npy_intp enc_stride0 = PyArray_STRIDE(py_encoded, 0);
    const npy_intp enc_stride1 = PyArray_STRIDE(py_encoded, 1);

    // Strides for nuc_factors: (K, 4) with float64
    const npy_intp nf_stride0 = PyArray_STRIDE(py_nuc_factors, 0);
    const npy_intp nf_stride1 = PyArray_STRIDE(py_nuc_factors, 1);

    // Strides for derivs: (N, K, 4) with float64
    const npy_intp d_stride0 = PyArray_STRIDE(py_derivs, 0);
    const npy_intp d_stride1 = PyArray_STRIDE(py_derivs, 1);
    const npy_intp d_stride2 = PyArray_STRIDE(py_derivs, 2);

    // Strides for spat_derivs: (N, B) with float64
    const npy_intp sd_stride0 = PyArray_STRIDE(py_spat_derivs, 0);
    const npy_intp sd_stride1 = PyArray_STRIDE(py_spat_derivs, 1);

    // Zero the outputs
    memset(derivs, 0, (size_t)(N * K * 4) * sizeof(double));
    memset(spat_derivs, 0, (size_t)(N * B) * sizeof(double));

    // Precompute spatial bin for each window position
    std::vector<int> spat_bins(num_wins);
    int center = (int)(B / 2);
    for (npy_intp w = 0; w < num_wins; ++w) {
        int b = (int)(w / spat_bin_size);
        if (symmetrize_spat && bidirect) {
            if (b > center) {
                b = b - (b - center) * 2;
            }
        }
        if (b < 0) b = 0;
        if (b >= B) b = (int)(B - 1);
        spat_bins[w] = b;
    }

    // Helper macros for strided access
    // encoded[seq, pos] => *(int8_t*)((char*)encoded_base + seq*enc_stride0 + pos*enc_stride1)
    // nuc_factors[d, nuc] => *(double*)((char*)nuc_factors_base + d*nf_stride0 + nuc*nf_stride1)
    // derivs[seq, d, nuc] => *(double*)((char*)derivs_base + seq*d_stride0 + d*d_stride1 + nuc*d_stride2)
    // spat_derivs[seq, b] => *(double*)((char*)spat_derivs_base + seq*sd_stride0 + b*sd_stride1)

    const char *encoded_base = (const char *)encoded;
    const char *nf_base = (const char *)nuc_factors;
    char *derivs_base = (char *)derivs;
    char *sd_base = (char *)spat_derivs;

    // We'll use contiguous data access for better performance.
    // Check if arrays are C-contiguous, and if so use direct indexing.
    // For now, use stride-based access for correctness.

    #pragma omp parallel for schedule(dynamic, 32)
    for (npy_intp seq = 0; seq < N; ++seq) {
        // Check train_mask
        if (!train_mask[seq]) continue;

        // Pointer offsets for this sequence
        const char *enc_seq = encoded_base + seq * enc_stride0;
        char *deriv_seq = derivs_base + seq * d_stride0;
        char *sd_seq = sd_base + seq * sd_stride0;

        // Forward strand
        for (npy_intp w = 0; w < num_wins; ++w) {
            // Compute product of nuc_factors for this window
            double product = 1.0;
            bool has_invalid = false;

            for (npy_intp d = 0; d < K; ++d) {
                int8_t base = *(const int8_t *)(enc_seq + (w + d) * enc_stride1);
                if (base < 0 || base > 3) {
                    has_invalid = true;
                    break;
                }
                double factor = *(const double *)(nf_base + d * nf_stride0 + base * nf_stride1);
                product *= factor;
            }

            if (has_invalid || product == 0.0) continue;

            // Accumulate spat_derivs (before spatial weighting)
            int sb = spat_bins[w];
            *(double *)(sd_seq + sb * sd_stride1) += product;

            // Spatial weighted product
            double weighted = product * spat_factors[sb];

            // Accumulate derivatives
            for (npy_intp d = 0; d < K; ++d) {
                int8_t base = *(const int8_t *)(enc_seq + (w + d) * enc_stride1);
                double factor_d = *(const double *)(nf_base + d * nf_stride0 + base * nf_stride1);
                double safe_factor = (factor_d > 0.0) ? factor_d : 1.0;
                double contrib = weighted / safe_factor;
                *(double *)(deriv_seq + d * d_stride1 + base * d_stride2) += contrib;
            }
        }

        // Reverse complement strand
        if (bidirect) {
            for (npy_intp w = 0; w < num_wins; ++w) {
                // RC: PSSM position K-1-d, complement of base at d
                double product = 1.0;
                bool has_invalid = false;

                for (npy_intp d = 0; d < K; ++d) {
                    int8_t base = *(const int8_t *)(enc_seq + (w + d) * enc_stride1);
                    if (base < 0 || base > 3) {
                        has_invalid = true;
                        break;
                    }
                    int comp = 3 - base;  // complement: A(0)<->T(3), C(1)<->G(2)
                    npy_intp pssm_pos = K - 1 - d;
                    double factor = *(const double *)(nf_base + pssm_pos * nf_stride0 + comp * nf_stride1);
                    product *= factor;
                }

                if (has_invalid || product == 0.0) continue;

                // Accumulate spat_derivs
                int sb = spat_bins[w];
                *(double *)(sd_seq + sb * sd_stride1) += product;

                // Spatial weighted product
                double weighted = product * spat_factors[sb];

                // Accumulate RC derivatives
                // deriv[seq, K-1-d, complement(base[d])] += weighted / factor_at_d
                for (npy_intp d = 0; d < K; ++d) {
                    int8_t base = *(const int8_t *)(enc_seq + (w + d) * enc_stride1);
                    int comp = 3 - base;
                    npy_intp pssm_pos = K - 1 - d;
                    double factor_d = *(const double *)(nf_base + pssm_pos * nf_stride0 + comp * nf_stride1);
                    double safe_factor = (factor_d > 0.0) ? factor_d : 1.0;
                    double contrib = weighted / safe_factor;
                    *(double *)(deriv_seq + pssm_pos * d_stride1 + comp * d_stride2) += contrib;
                }
            }
        }
    }

    Py_RETURN_NONE;
}


// ---------------------------------------------------------------------------
// pyprego_batch_extract_energies
// ---------------------------------------------------------------------------
// Python signature:
//   batch_extract_energies(encoded, log_pssm_list, spat_factors_list,
//                          spat_bin_sizes, bidirect, output) -> None
//
// Parameters:
//   encoded          : ndarray[int8, (N, L)]       - pre-encoded sequences
//   log_pssm_list    : Python list of M ndarrays, each (K_m, 4) float64
//   spat_factors_list: Python list of M ndarrays, each (B_m,) float64 (RAW, not log)
//   spat_bin_sizes   : ndarray[int32, (M,)]        - bin size per motif
//   bidirect         : int                          - 1 if bidirectional
//   output           : ndarray[float64, (N, M)]     - pre-allocated output (filled in-place)

PyObject *pyprego_batch_extract_energies(PyObject * /*self*/, PyObject *args)
{
    PyArrayObject *py_encoded = nullptr;
    PyObject *py_log_pssm_list = nullptr;
    PyObject *py_spat_factors_list = nullptr;
    PyArrayObject *py_spat_bin_sizes = nullptr;
    int bidirect = 0;
    PyArrayObject *py_output = nullptr;

    if (!PyArg_ParseTuple(args, "O!OOO!iO!",
                          &PyArray_Type, &py_encoded,
                          &py_log_pssm_list,
                          &py_spat_factors_list,
                          &PyArray_Type, &py_spat_bin_sizes,
                          &bidirect,
                          &PyArray_Type, &py_output))
        return NULL;

    // Validate encoded
    if (PyArray_NDIM(py_encoded) != 2) {
        PyErr_SetString(PyExc_ValueError, "encoded must be 2-D (N, L)");
        return NULL;
    }
    if (PyArray_NDIM(py_output) != 2) {
        PyErr_SetString(PyExc_ValueError, "output must be 2-D (N, M)");
        return NULL;
    }
    if (!PyList_Check(py_log_pssm_list)) {
        PyErr_SetString(PyExc_TypeError, "log_pssm_list must be a Python list");
        return NULL;
    }
    if (!PyList_Check(py_spat_factors_list)) {
        PyErr_SetString(PyExc_TypeError, "spat_factors_list must be a Python list");
        return NULL;
    }

    const npy_intp N = PyArray_DIM(py_encoded, 0);
    const npy_intp L = PyArray_DIM(py_encoded, 1);
    const Py_ssize_t M = PyList_Size(py_log_pssm_list);

    if (PyList_Size(py_spat_factors_list) != M) {
        PyErr_SetString(PyExc_ValueError, "spat_factors_list length must match log_pssm_list length");
        return NULL;
    }
    if (PyArray_DIM(py_output, 0) != N || PyArray_DIM(py_output, 1) != M) {
        PyErr_SetString(PyExc_ValueError, "output shape must be (N, M)");
        return NULL;
    }

    // ---- Unpack all motif data into C++ vectors for thread-safe access ----

    // Per-motif: log_pssm pointer, K, avg_log_prob per position,
    //            spat_factors pointer (raw), n_bins, bin_size
    struct MotifInfo {
        const double *log_pssm;    // (K, 4) row-major
        npy_intp K;
        std::vector<double> avg_log_prob;  // (K,) mean of log_pssm row
        const double *spat_factors;        // raw spatial factors
        npy_intp n_bins;
        int bin_size;
    };

    std::vector<MotifInfo> motifs(M);
    // Keep references to borrowed arrays to prevent GC during computation
    // (The list items are borrowed references, but the list itself is alive
    //  for the duration of this call, so this is safe.)

    const int *spat_bin_sizes_ptr = (const int *)PyArray_DATA(py_spat_bin_sizes);

    for (Py_ssize_t m = 0; m < M; ++m) {
        PyArrayObject *pssm_arr = (PyArrayObject *)PyList_GET_ITEM(py_log_pssm_list, m);
        PyArrayObject *spat_arr = (PyArrayObject *)PyList_GET_ITEM(py_spat_factors_list, m);

        if (PyArray_NDIM(pssm_arr) != 2 || PyArray_DIM(pssm_arr, 1) != 4) {
            PyErr_Format(PyExc_ValueError,
                "log_pssm_list[%zd] must be 2-D with shape (K, 4)", m);
            return NULL;
        }
        if (PyArray_NDIM(spat_arr) != 1) {
            PyErr_Format(PyExc_ValueError,
                "spat_factors_list[%zd] must be 1-D", m);
            return NULL;
        }

        MotifInfo &mi = motifs[m];
        mi.log_pssm = (const double *)PyArray_DATA(pssm_arr);
        mi.K = PyArray_DIM(pssm_arr, 0);
        mi.spat_factors = (const double *)PyArray_DATA(spat_arr);
        mi.n_bins = PyArray_DIM(spat_arr, 0);
        mi.bin_size = spat_bin_sizes_ptr[m];

        // Precompute avg_log_prob per position (for N-base handling)
        mi.avg_log_prob.resize(mi.K);
        for (npy_intp d = 0; d < mi.K; ++d) {
            double sum = 0.0;
            for (int nuc = 0; nuc < 4; ++nuc) {
                sum += mi.log_pssm[d * 4 + nuc];
            }
            mi.avg_log_prob[d] = sum / 4.0;
        }
    }

    // Get encoded data pointer and strides
    const int8_t *encoded = (const int8_t *)PyArray_DATA(py_encoded);
    const npy_intp enc_stride0 = PyArray_STRIDE(py_encoded, 0);
    const npy_intp enc_stride1 = PyArray_STRIDE(py_encoded, 1);

    // Output: (N, M) float64, C-contiguous expected but use strides to be safe
    double *output = (double *)PyArray_DATA(py_output);
    const npy_intp out_stride0 = PyArray_STRIDE(py_output, 0);
    const npy_intp out_stride1 = PyArray_STRIDE(py_output, 1);

    const char *enc_base = (const char *)encoded;
    char *out_base = (char *)output;

    // ---- Main computation: parallel over sequences ----
    #pragma omp parallel for schedule(dynamic, 64)
    for (npy_intp seq = 0; seq < N; ++seq) {
        const char *enc_seq = enc_base + seq * enc_stride0;

        for (Py_ssize_t m = 0; m < M; ++m) {
            const MotifInfo &mi = motifs[m];
            const npy_intp K = mi.K;
            const npy_intp num_wins = L - K + 1;

            if (num_wins <= 0) {
                *(double *)(out_base + seq * out_stride0 + m * out_stride1) = -std::numeric_limits<double>::infinity();
                continue;
            }

            // Allocate scores buffer: up to num_wins * (bidirect ? 2 : 1) entries
            const int n_scores_max = (int)(num_wins * (bidirect ? 2 : 1));
            // Use thread-local stack allocation for small motifs, heap for large
            std::vector<double> scores_buf(n_scores_max);
            int n_scores = 0;

            // Precompute spatial log factors and bin mapping for this motif
            // (done per-motif, could be precomputed outside the seq loop but
            //  the cost is negligible compared to window scoring)

            for (npy_intp w = 0; w < num_wins; ++w) {
                // Spatial bin
                int sb = (int)(w / mi.bin_size);
                if (sb >= mi.n_bins) sb = (int)(mi.n_bins - 1);
                if (sb < 0) sb = 0;
                double log_spat = std::log(mi.spat_factors[sb]);

                // ---- Forward strand ----
                double log_score = 0.0;
                for (npy_intp d = 0; d < K; ++d) {
                    int8_t base = *(const int8_t *)(enc_seq + (w + d) * enc_stride1);
                    if (base >= 0 && base <= 3) {
                        log_score += mi.log_pssm[d * 4 + base];
                    } else {
                        // N base: use average log prob
                        log_score += mi.avg_log_prob[d];
                    }
                }
                log_score += log_spat;
                scores_buf[n_scores++] = log_score;

                // ---- Reverse complement ----
                if (bidirect) {
                    double rc_log_score = 0.0;
                    for (npy_intp d = 0; d < K; ++d) {
                        int8_t base = *(const int8_t *)(enc_seq + (w + d) * enc_stride1);
                        if (base >= 0 && base <= 3) {
                            int comp = 3 - base;
                            npy_intp pssm_pos = K - 1 - d;
                            rc_log_score += mi.log_pssm[pssm_pos * 4 + comp];
                        } else {
                            // N base: use average log prob at RC position
                            npy_intp pssm_pos = K - 1 - d;
                            rc_log_score += mi.avg_log_prob[pssm_pos];
                        }
                    }
                    rc_log_score += log_spat;
                    scores_buf[n_scores++] = rc_log_score;
                }
            }

            // logSumExp over all scores
            double result;
            if (n_scores == 0) {
                result = -std::numeric_limits<double>::infinity();
            } else {
                result = log_sum_exp(scores_buf.data(), n_scores);
            }

            *(double *)(out_base + seq * out_stride0 + m * out_stride1) = result;
        }
    }

    Py_RETURN_NONE;
}
