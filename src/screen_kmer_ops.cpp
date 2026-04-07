// screen_kmer_ops.cpp -- Fast k-mer screening with Pearson correlation.
//
// Provides pyprego_screen_kmers: count all k-mers (pure and gapped) across
// sequences and compute Pearson correlation with one or more response
// variables. Filters results by r^2 threshold.
//
// Supports gapped k-mers (wildcard N positions) via min_gap/max_gap params.
// For gap_len g at gap_pos p in a k-mer of length k, the fixed (non-N)
// positions determine a base-4 hash of length k-g. Each (gap_len, gap_pos)
// pattern has its own block of 4^(k-g) accumulators.
//
// Memory: O(n_threads × n_total_kmers) where n_total_kmers =
//   4^k + sum_{g=min_gap..max_gap, g>0} (k-g+1) × 4^(k-g)
// For k=8, max_gap=1: 196,608 kmers × ~24 bytes/kmer × n_threads ≈ 5 MB/thread

#include "_pyprego.h"
#include "dna_utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

static const char BASE4_CHARS[4] = {'A', 'C', 'G', 'T'};

static std::string hash_to_kmer(int hash, int k)
{
    std::string km(k, 'A');
    for (int pos = k - 1; pos >= 0; --pos) {
        km[pos] = BASE4_CHARS[hash & 3];
        hash >>= 2;
    }
    return km;
}

// Build the gapped kmer string from (gap_len, gap_pos, fixed_hash, k).
// E.g. k=8, gap_len=1, gap_pos=3, fixed_hash encodes "ACGTACG"
// -> "ACGNACGT" (N at position 3, fixed bases around it)
static std::string gapped_kmer_name(int gap_len, int gap_pos, int fixed_hash, int k)
{
    int n_fixed = k - gap_len;
    // Decode fixed_hash into n_fixed bases
    std::string fixed(n_fixed, 'A');
    for (int i = n_fixed - 1; i >= 0; --i) {
        fixed[i] = BASE4_CHARS[fixed_hash & 3];
        fixed_hash >>= 2;
    }
    // Build full kmer with N's at gap positions
    std::string km(k, 'N');
    int fi = 0;
    for (int i = 0; i < k; ++i) {
        if (i >= gap_pos && i < gap_pos + gap_len) {
            // gap position, already 'N'
        } else {
            km[i] = fixed[fi++];
        }
    }
    return km;
}

// Describe the layout of gapped kmer accumulators.
// For each gap pattern (gap_len, gap_pos), we store:
//   offset: starting index in the flat accumulator arrays
//   n_kmers: 4^(k-gap_len)
struct GapPattern {
    int gap_len;
    int gap_pos;
    int n_fixed;    // k - gap_len
    int n_kmers;    // 4^n_fixed
    int offset;     // global offset in accumulator arrays
};

// ---------------------------------------------------------------------------
// pyprego_screen_kmers
// ---------------------------------------------------------------------------
// Python signature:
//   screen_kmers(sequences, response, kmer_length, min_cor, min_gap=0, max_gap=0)
//       -> (kmer_names: list[str], max_r2: ndarray, correlations: ndarray 2D,
//           avg_n: ndarray, avg_var: ndarray)
//
// sequences: int8 array (n_seq, seq_len) -- encoded (A=0,C=1,G=2,T=3,N=-1)
// response:  float64 array (n_seq, n_resp)
// kmer_length: int
// min_cor: double (minimum correlation threshold; filter by r^2 >= min_cor^2)
// min_gap: int (default 0)
// max_gap: int (default 0)

PyObject *pyprego_screen_kmers(PyObject * /*self*/, PyObject *args)
{
    PyArrayObject *py_sequences = nullptr;
    PyArrayObject *py_response = nullptr;
    int kmer_length = 0;
    double min_cor = 0.0;
    int min_gap = 0;
    int max_gap = 0;

    if (!PyArg_ParseTuple(args, "O!O!id|ii",
                          &PyArray_Type, &py_sequences,
                          &PyArray_Type, &py_response,
                          &kmer_length, &min_cor,
                          &min_gap, &max_gap))
        return NULL;

    // Validate sequences array
    if (!PPIS2D(py_sequences)) {
        PyErr_SetString(PyExc_ValueError, "sequences must be a 2D array");
        return NULL;
    }
    if (PyArray_TYPE(py_sequences) != NPY_INT8) {
        PyErr_SetString(PyExc_TypeError, "sequences must be int8");
        return NULL;
    }
    if (!PyArray_IS_C_CONTIGUOUS(py_sequences)) {
        PyErr_SetString(PyExc_ValueError, "sequences must be C-contiguous");
        return NULL;
    }

    // Validate response array
    if (!PPIS2D(py_response)) {
        PyErr_SetString(PyExc_ValueError, "response must be a 2D array");
        return NULL;
    }
    if (PyArray_TYPE(py_response) != NPY_FLOAT64) {
        PyErr_SetString(PyExc_TypeError, "response must be float64");
        return NULL;
    }
    if (!PyArray_IS_C_CONTIGUOUS(py_response)) {
        PyErr_SetString(PyExc_ValueError, "response must be C-contiguous");
        return NULL;
    }

    if (kmer_length < 1) {
        PyErr_SetString(PyExc_ValueError, "kmer_length must be >= 1");
        return NULL;
    }
    if (min_gap < 0 || max_gap < 0 || max_gap > kmer_length || max_gap < min_gap) {
        PyErr_SetString(PyExc_ValueError,
                        "Invalid gap parameters: need 0 <= min_gap <= max_gap <= kmer_length");
        return NULL;
    }

    const npy_intp n_seq = PyArray_DIM(py_sequences, 0);
    const npy_intp seq_len = PyArray_DIM(py_sequences, 1);
    const npy_intp n_resp = PyArray_DIM(py_response, 1);

    if (PyArray_DIM(py_response, 0) != n_seq) {
        PyErr_SetString(PyExc_ValueError,
                        "sequences and response must have the same number of rows");
        return NULL;
    }

    // ====================================================================
    // Build kmer layout: pure kmers + gapped patterns
    // ====================================================================
    int n_pure = 1;
    for (int i = 0; i < kmer_length; ++i) {
        n_pure *= 4;
        if (n_pure > (1 << 24)) {
            PyErr_SetString(PyExc_ValueError,
                            "kmer_length too large (max 12 for screen_kmers)");
            return NULL;
        }
    }

    // Build gap patterns
    std::vector<GapPattern> gap_patterns;
    int n_gapped_total = 0;
    for (int g = (min_gap > 0 ? min_gap : 1); g <= max_gap; ++g) {
        for (int p = 0; p <= kmer_length - g; ++p) {
            GapPattern gp;
            gp.gap_len = g;
            gp.gap_pos = p;
            gp.n_fixed = kmer_length - g;
            gp.n_kmers = 1;
            for (int i = 0; i < gp.n_fixed; ++i) gp.n_kmers *= 4;
            gp.offset = n_pure + n_gapped_total;
            gap_patterns.push_back(gp);
            n_gapped_total += gp.n_kmers;
        }
    }

    // If min_gap > 0, we skip pure kmers in the output
    bool include_pure = (min_gap == 0);
    int n_total = (include_pure ? n_pure : 0) + n_gapped_total;

    // Adjust offsets if we exclude pure kmers
    if (!include_pure) {
        for (auto &gp : gap_patterns) {
            gp.offset -= n_pure;
        }
    }

    if (n_total == 0) {
        // Nothing to screen
        PyObject *name_list = PyList_New(0);
        npy_intp dim0[1] = {0};
        npy_intp dim2d[2] = {0, n_resp};
        PyObject *result = PyTuple_Pack(5,
            name_list,
            (PyObject *)PyArray_SimpleNew(1, dim0, NPY_FLOAT64),
            (PyObject *)PyArray_SimpleNew(2, dim2d, NPY_FLOAT64),
            (PyObject *)PyArray_SimpleNew(1, dim0, NPY_FLOAT64),
            (PyObject *)PyArray_SimpleNew(1, dim0, NPY_FLOAT64));
        Py_DECREF(name_list);
        return result;
    }

    const int8_t *seq_data = (const int8_t *)PyArray_DATA(py_sequences);
    const double *resp_data = (const double *)PyArray_DATA(py_response);
    const int pure_mask = n_pure - 1;

    // ====================================================================
    // Response statistics: mean and variance per response column
    // ====================================================================
    std::vector<double> resp_mean(n_resp, 0.0);
    std::vector<double> resp_var(n_resp, 0.0);
    for (npy_intp ri = 0; ri < n_resp; ++ri) {
        double sum = 0.0, sum2 = 0.0;
        for (npy_intp si = 0; si < n_seq; ++si) {
            double v = resp_data[si * n_resp + ri];
            sum += v;
            sum2 += v * v;
        }
        resp_mean[ri] = sum / n_seq;
        resp_var[ri] = sum2 / n_seq - resp_mean[ri] * resp_mean[ri];
    }

    // ====================================================================
    // Per-thread accumulators
    // Memory: n_threads × n_total × (2 + n_resp) doubles + n_total ints scratch
    // ====================================================================
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif

    std::vector<double> sum_count(n_total, 0.0);
    std::vector<double> sum_count2(n_total, 0.0);
    std::vector<double> cross_prod((size_t)n_total * n_resp, 0.0);

    std::vector<std::vector<double>> thr_sum_count(n_threads, std::vector<double>(n_total, 0.0));
    std::vector<std::vector<double>> thr_sum_count2(n_threads, std::vector<double>(n_total, 0.0));
    std::vector<std::vector<double>> thr_cross_prod(n_threads, std::vector<double>((size_t)n_total * n_resp, 0.0));
    std::vector<std::vector<int32_t>> thr_scratch(n_threads, std::vector<int32_t>(n_total, 0));

    #pragma omp parallel
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        int32_t *scratch = thr_scratch[tid].data();
        double *t_sum = thr_sum_count[tid].data();
        double *t_sum2 = thr_sum_count2[tid].data();
        double *t_cross = thr_cross_prod[tid].data();

        #pragma omp for schedule(dynamic, 64)
        for (npy_intp si = 0; si < n_seq; ++si) {
            const int8_t *seq = seq_data + si * seq_len;
            const double *resp_row = resp_data + si * n_resp;
            int num_wins = (int)seq_len - kmer_length + 1;
            if (num_wins <= 0) continue;

            // ── Pure k-mers: rolling base-4 hash ──
            if (include_pure) {
                int h = 0;
                bool valid = true;
                for (int j = 0; j < kmer_length; ++j) {
                    int b = seq[j];
                    if (b < 0 || b > 3) { valid = false; break; }
                    h = (h << 2) | b;
                }
                if (valid) scratch[h]++;

                for (int w = 1; w < num_wins; ++w) {
                    int b_new = seq[w + kmer_length - 1];
                    if (b_new < 0 || b_new > 3) {
                        valid = false;
                        continue;
                    }
                    if (!valid) {
                        h = 0;
                        valid = true;
                        for (int j = 0; j < kmer_length; ++j) {
                            int b = seq[w + j];
                            if (b < 0 || b > 3) { valid = false; break; }
                            h = (h << 2) | b;
                        }
                        if (valid) scratch[h]++;
                    } else {
                        h = ((h << 2) | b_new) & pure_mask;
                        scratch[h]++;
                    }
                }
            }

            // ── Gapped k-mers ──
            // For each window and each gap pattern, compute the fixed-position
            // hash and increment the corresponding counter.
            if (!gap_patterns.empty()) {
                for (int w = 0; w < num_wins; ++w) {
                    const int8_t *win = seq + w;

                    for (size_t gi = 0; gi < gap_patterns.size(); ++gi) {
                        const GapPattern &gp = gap_patterns[gi];

                        // Compute hash from fixed (non-gap) positions
                        int fh = 0;
                        bool fvalid = true;
                        for (int pos = 0; pos < kmer_length; ++pos) {
                            if (pos >= gp.gap_pos && pos < gp.gap_pos + gp.gap_len)
                                continue; // skip gap positions
                            int b = win[pos];
                            if (b < 0 || b > 3) { fvalid = false; break; }
                            fh = (fh << 2) | b;
                        }
                        if (fvalid) {
                            scratch[gp.offset + fh]++;
                        }
                    }
                }
            }

            // Accumulate statistics from this sequence and clear scratch
            for (int ki = 0; ki < n_total; ++ki) {
                int32_t c = scratch[ki];
                if (c == 0) continue;
                double cd = (double)c;
                t_sum[ki] += cd;
                t_sum2[ki] += cd * cd;
                double *cp = t_cross + (size_t)ki * n_resp;
                for (npy_intp ri = 0; ri < n_resp; ++ri) {
                    cp[ri] += cd * resp_row[ri];
                }
                scratch[ki] = 0;
            }
        }
    }

    // Merge per-thread accumulators
    for (int t = 0; t < n_threads; ++t) {
        const double *t_sum = thr_sum_count[t].data();
        const double *t_sum2 = thr_sum_count2[t].data();
        const double *t_cross = thr_cross_prod[t].data();
        for (int ki = 0; ki < n_total; ++ki) {
            sum_count[ki] += t_sum[ki];
            sum_count2[ki] += t_sum2[ki];
        }
        for (size_t i = 0; i < (size_t)n_total * n_resp; ++i) {
            cross_prod[i] += t_cross[i];
        }
    }

    // ====================================================================
    // Compute correlations and filter
    // ====================================================================
    std::vector<double> avg_n(n_total);
    std::vector<double> avg_var(n_total);
    std::vector<double> corr_matrix((size_t)n_total * n_resp, 0.0);
    std::vector<double> max_r2(n_total, 0.0);
    double r2_thresh = min_cor * min_cor;

    for (int ki = 0; ki < n_total; ++ki) {
        double mean_c = sum_count[ki] / n_seq;
        double var_c = sum_count2[ki] / n_seq - mean_c * mean_c;
        avg_n[ki] = mean_c;
        avg_var[ki] = var_c;

        if (var_c < 1e-15) {
            max_r2[ki] = 0.0;
            continue;
        }

        double best_r2 = 0.0;
        const double *cp = cross_prod.data() + (size_t)ki * n_resp;
        for (npy_intp ri = 0; ri < n_resp; ++ri) {
            double cov_val = cp[ri] / n_seq - mean_c * resp_mean[ri];
            double denom = std::sqrt(var_c * resp_var[ri]);
            double r = (denom > 1e-15) ? (cov_val / denom) : 0.0;
            corr_matrix[ki * n_resp + ri] = r;
            double r2 = r * r;
            if (r2 > best_r2) best_r2 = r2;
        }
        max_r2[ki] = best_r2;
    }

    // Filter
    std::vector<int> valid_indices;
    valid_indices.reserve(n_total);
    for (int ki = 0; ki < n_total; ++ki) {
        if (avg_var[ki] < 1e-15) continue;
        if (min_cor > 0 && max_r2[ki] < r2_thresh) continue;
        valid_indices.push_back(ki);
    }
    int n_valid = (int)valid_indices.size();

    // ====================================================================
    // Build kmer name for each valid index
    // ====================================================================
    // Helper: map global index -> kmer name string
    auto index_to_name = [&](int idx) -> std::string {
        if (include_pure && idx < n_pure) {
            return hash_to_kmer(idx, kmer_length);
        }
        // Find which gap pattern this index belongs to
        for (const auto &gp : gap_patterns) {
            if (idx >= gp.offset && idx < gp.offset + gp.n_kmers) {
                int local_hash = idx - gp.offset;
                return gapped_kmer_name(gp.gap_len, gp.gap_pos, local_hash, kmer_length);
            }
        }
        return "???"; // should never happen
    };

    // ====================================================================
    // Build return values
    // ====================================================================
    PyObject *name_list = PyList_New(n_valid);
    if (!name_list) return NULL;
    for (int i = 0; i < n_valid; ++i) {
        std::string km = index_to_name(valid_indices[i]);
        PyObject *s = PyUnicode_FromStringAndSize(km.c_str(), km.size());
        if (!s) { Py_DECREF(name_list); return NULL; }
        PyList_SET_ITEM(name_list, i, s);
    }

    npy_intp dim1[1] = {(npy_intp)n_valid};

    PyArrayObject *py_max_r2 = (PyArrayObject *)PyArray_SimpleNew(1, dim1, NPY_FLOAT64);
    if (!py_max_r2) { Py_DECREF(name_list); return NULL; }
    {
        double *out = (double *)PyArray_DATA(py_max_r2);
        for (int i = 0; i < n_valid; ++i) out[i] = max_r2[valid_indices[i]];
    }

    npy_intp dim2[2] = {(npy_intp)n_valid, (npy_intp)n_resp};
    PyArrayObject *py_corr = (PyArrayObject *)PyArray_SimpleNew(2, dim2, NPY_FLOAT64);
    if (!py_corr) { Py_DECREF(name_list); Py_DECREF(py_max_r2); return NULL; }
    {
        double *out = (double *)PyArray_DATA(py_corr);
        for (int i = 0; i < n_valid; ++i) {
            int ki = valid_indices[i];
            for (npy_intp ri = 0; ri < n_resp; ++ri)
                out[i * n_resp + ri] = corr_matrix[ki * n_resp + ri];
        }
    }

    PyArrayObject *py_avg_n = (PyArrayObject *)PyArray_SimpleNew(1, dim1, NPY_FLOAT64);
    if (!py_avg_n) { Py_DECREF(name_list); Py_DECREF(py_max_r2); Py_DECREF(py_corr); return NULL; }
    {
        double *out = (double *)PyArray_DATA(py_avg_n);
        for (int i = 0; i < n_valid; ++i) out[i] = avg_n[valid_indices[i]];
    }

    PyArrayObject *py_avg_var = (PyArrayObject *)PyArray_SimpleNew(1, dim1, NPY_FLOAT64);
    if (!py_avg_var) { Py_DECREF(name_list); Py_DECREF(py_max_r2); Py_DECREF(py_corr); Py_DECREF(py_avg_n); return NULL; }
    {
        double *out = (double *)PyArray_DATA(py_avg_var);
        for (int i = 0; i < n_valid; ++i) out[i] = avg_var[valid_indices[i]];
    }

    PyObject *result = PyTuple_Pack(5,
                                    name_list,
                                    (PyObject *)py_max_r2,
                                    (PyObject *)py_corr,
                                    (PyObject *)py_avg_n,
                                    (PyObject *)py_avg_var);
    Py_DECREF(name_list);
    Py_DECREF(py_max_r2);
    Py_DECREF(py_corr);
    Py_DECREF(py_avg_n);
    Py_DECREF(py_avg_var);

    return result;
}
