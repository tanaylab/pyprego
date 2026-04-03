// kmer_ops.cpp -- Fast k-mer enumeration kernels.
//
// Provides pyprego_kmer_matrix: a C++ implementation of kmer_matrix()
// with OpenMP parallelism over sequences.

#include "_pyprego.h"
#include "dna_utils.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Base-4 characters for k-mer string generation: A=0, C=1, G=2, T=3
static const char BASE4_CHARS[4] = {'A', 'C', 'G', 'T'};

// Generate all 4^k pure k-mer strings in lexicographic (base-4) order.
// index 0 = "AAA...", index 1 = "AAA...C", etc.
static std::vector<std::string> generate_all_pure_kmers(int k)
{
    int n_kmers = 1;
    for (int i = 0; i < k; ++i) n_kmers *= 4;

    std::vector<std::string> kmers(n_kmers);
    for (int idx = 0; idx < n_kmers; ++idx) {
        std::string km(k, 'A');
        int val = idx;
        for (int pos = k - 1; pos >= 0; --pos) {
            km[pos] = BASE4_CHARS[val & 3];
            val >>= 2;
        }
        kmers[idx] = km;
    }
    return kmers;
}

// Compute base-4 hash for a k-mer string (pure, no N).
// Returns -1 if any character is not ACGT.
static int kmer_hash(const char *s, int k)
{
    int h = 0;
    for (int i = 0; i < k; ++i) {
        int b = encode_char(s[i]);
        if (b < 0) return -1;
        h = (h << 2) | b;
    }
    return h;
}

// ---------------------------------------------------------------------------
// pyprego_kmer_matrix
// ---------------------------------------------------------------------------
// Python signature:
//   kmer_matrix_c(sequences: list[str], k: int, max_gap: int = 0) -> tuple[ndarray, list[str]]
//
// For pure k-mers (max_gap == 0):
//   - Returns counts for ALL 4^k k-mers in base-4 order.
//   - Uses sliding window + base-4 integer hashing.
//
// For gapped k-mers (max_gap > 0):
//   - Includes pure k-mers PLUS gapped variants (N at gap positions).
//   - Matches the R KmerRegression.cpp algorithm:
//     for each window of length k, enumerate all gap sizes 1..max_gap,
//     and at each gap size g, enumerate all starting positions 0..k-g,
//     replacing g consecutive characters with 'N'.
//   - Collects unique k-mer names across all sequences.

PyObject *pyprego_kmer_matrix(PyObject * /*self*/, PyObject *args)
{
    PyObject *seq_list;
    int k;
    int max_gap = 0;

    if (!PyArg_ParseTuple(args, "Oi|i", &seq_list, &k, &max_gap))
        return NULL;

    if (!PyList_Check(seq_list)) {
        PyErr_SetString(PyExc_TypeError, "sequences must be a list of strings");
        return NULL;
    }

    const Py_ssize_t n_seqs = PyList_GET_SIZE(seq_list);
    if (k < 1) {
        PyErr_SetString(PyExc_ValueError, "k must be >= 1");
        return NULL;
    }
    if (max_gap < 0) {
        PyErr_SetString(PyExc_ValueError, "max_gap must be >= 0");
        return NULL;
    }

    // ── Extract sequences as std::string ──
    std::vector<std::string> seqs(n_seqs);
    for (Py_ssize_t i = 0; i < n_seqs; ++i) {
        PyObject *item = PyList_GET_ITEM(seq_list, i);
        if (!PyUnicode_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "All sequences must be strings");
            return NULL;
        }
        Py_ssize_t len;
        const char *data = PyUnicode_AsUTF8AndSize(item, &len);
        if (!data) return NULL;
        seqs[i].assign(data, (size_t)len);
        // uppercase
        for (auto &ch : seqs[i]) {
            if (ch >= 'a' && ch <= 'z') ch -= 32;
        }
    }

    // ====================================================================
    // PURE K-MER PATH (no gaps)
    // ====================================================================
    if (max_gap == 0) {
        int n_kmers = 1;
        for (int i = 0; i < k; ++i) n_kmers *= 4;

        // Allocate output array (n_seqs, n_kmers)
        npy_intp dims[2] = {(npy_intp)n_seqs, (npy_intp)n_kmers};
        PyArrayObject *arr = (PyArrayObject *)PyArray_ZEROS(2, dims, NPY_INT32, 0);
        if (!arr) return NULL;

        int32_t *data = (int32_t *)PyArray_DATA(arr);

        // Parallel over sequences
        #pragma omp parallel for schedule(dynamic, 64)
        for (Py_ssize_t si = 0; si < n_seqs; ++si) {
            const std::string &seq = seqs[si];
            int slen = (int)seq.size();
            int num_wins = slen - k + 1;
            if (num_wins <= 0) continue;

            int32_t *row = data + si * n_kmers;
            const char *s = seq.c_str();

            // Compute first window hash
            int h = 0;
            bool valid = true;
            for (int j = 0; j < k; ++j) {
                int b = encode_char(s[j]);
                if (b < 0) { valid = false; break; }
                h = (h << 2) | b;
            }
            int mask = n_kmers - 1;  // 4^k - 1

            if (valid) {
                row[h]++;
            }

            // Slide the window
            for (int w = 1; w < num_wins; ++w) {
                int b_new = encode_char(s[w + k - 1]);
                int b_old = encode_char(s[w - 1]);

                if (b_new < 0 || b_old < 0) {
                    // Recompute hash from scratch
                    h = 0;
                    valid = true;
                    for (int j = 0; j < k; ++j) {
                        int b = encode_char(s[w + j]);
                        if (b < 0) { valid = false; break; }
                        h = (h << 2) | b;
                    }
                    if (valid) row[h]++;
                } else if (!valid) {
                    // Previous window was invalid; recompute
                    h = 0;
                    valid = true;
                    for (int j = 0; j < k; ++j) {
                        int b = encode_char(s[w + j]);
                        if (b < 0) { valid = false; break; }
                        h = (h << 2) | b;
                    }
                    if (valid) row[h]++;
                } else {
                    // Rolling hash update
                    h = ((h << 2) | b_new) & mask;
                    row[h]++;
                }
            }
        }

        // Build k-mer name list
        std::vector<std::string> kmer_names = generate_all_pure_kmers(k);
        PyObject *name_list = PyList_New(n_kmers);
        if (!name_list) { Py_DECREF(arr); return NULL; }
        for (int i = 0; i < n_kmers; ++i) {
            PyObject *s = PyUnicode_FromStringAndSize(kmer_names[i].c_str(), k);
            if (!s) { Py_DECREF(arr); Py_DECREF(name_list); return NULL; }
            PyList_SET_ITEM(name_list, i, s);  // steals ref
        }

        PyObject *result = PyTuple_Pack(2, (PyObject *)arr, name_list);
        Py_DECREF(arr);
        Py_DECREF(name_list);
        return result;
    }

    // ====================================================================
    // GAPPED K-MER PATH (max_gap > 0)
    // ====================================================================
    // For each sequence, for each sliding window of length k:
    //   1. Record the pure k-mer (unless it contains non-ACGT)
    //   2. For gap sizes g = 1..max_gap, for each starting position pos = 0..k-g:
    //      replace positions pos..pos+g-1 with 'N', record the gapped k-mer
    //
    // This matches R's KmerRegression.cpp approach.

    // Per-sequence maps (parallel)
    std::vector<std::unordered_map<std::string, int32_t>> per_seq_maps(n_seqs);

    #pragma omp parallel for schedule(dynamic, 64)
    for (Py_ssize_t si = 0; si < n_seqs; ++si) {
        const std::string &seq = seqs[si];
        int slen = (int)seq.size();
        int num_wins = slen - k + 1;
        if (num_wins <= 0) continue;

        auto &smap = per_seq_maps[si];
        std::string kmer_buf(k, 'A');

        for (int w = 0; w < num_wins; ++w) {
            const char *win = seq.c_str() + w;

            // Check for valid bases and build pure k-mer
            bool has_n = false;
            for (int j = 0; j < k; ++j) {
                kmer_buf[j] = win[j];
                if (encode_char(win[j]) < 0) has_n = true;
            }

            if (!has_n) {
                smap[kmer_buf]++;
            }

            // Gapped variants
            for (int g = 1; g <= max_gap; ++g) {
                for (int pos = 0; pos <= k - g; ++pos) {
                    // Build gapped k-mer: copy window, replace pos..pos+g-1 with 'N'
                    // Check that non-gap positions are valid bases
                    bool gap_valid = true;
                    for (int j = 0; j < k; ++j) {
                        if (j >= pos && j < pos + g) {
                            kmer_buf[j] = 'N';
                        } else {
                            int b = encode_char(win[j]);
                            if (b < 0) { gap_valid = false; break; }
                            kmer_buf[j] = BASE4_CHARS[b];
                        }
                    }
                    if (gap_valid) {
                        smap[kmer_buf]++;
                    }
                }
            }
        }
    }

    // Collect all unique k-mer names and assign column indices.
    // Use the SAME ordering as the Python generate_kmers() function:
    //   1. All pure k-mers in base-4 order (4^k of them)
    //   2. For gap g = 1..max_gap, for pos = 0..k-g:
    //        for each pure k-mer pattern, the gapped variant at (g, pos)
    //      But unique only (dict.fromkeys preserves first occurrence).
    //
    // To match Python exactly, we generate the same ordered list.

    // Step 1: Generate pure k-mers
    std::vector<std::string> all_pure = generate_all_pure_kmers(k);

    // Step 2: Generate gapped k-mers in same order as Python's generate_kmers
    // Python code:
    //   for g in range(min_gap, max_gap+1):
    //       if g == 0: continue
    //       gap_str = "N" * g
    //       for pos in range(k - g + 1):
    //           for km in base_kmers:
    //               gapped = km[:pos] + gap_str + km[pos+g:]
    //               gap_kmers.append(gapped)
    //   return list(dict.fromkeys(base_kmers + gap_kmers))

    // We need dict.fromkeys(base_kmers + gap_kmers) to get unique ordered names
    std::vector<std::string> ordered_names;
    std::unordered_map<std::string, int> name_to_col;

    // First add pure k-mers
    for (auto &km : all_pure) {
        if (name_to_col.find(km) == name_to_col.end()) {
            name_to_col[km] = (int)ordered_names.size();
            ordered_names.push_back(km);
        }
    }

    // Then add gapped k-mers in Python's order
    for (int g = 1; g <= max_gap; ++g) {
        std::string gap_str(g, 'N');
        for (int pos = 0; pos <= k - g; ++pos) {
            for (auto &km : all_pure) {
                std::string gapped = km.substr(0, pos) + gap_str + km.substr(pos + g);
                if (name_to_col.find(gapped) == name_to_col.end()) {
                    name_to_col[gapped] = (int)ordered_names.size();
                    ordered_names.push_back(gapped);
                }
            }
        }
    }

    int n_cols = (int)ordered_names.size();

    // Build output array
    npy_intp dims[2] = {(npy_intp)n_seqs, (npy_intp)n_cols};
    PyArrayObject *arr = (PyArrayObject *)PyArray_ZEROS(2, dims, NPY_INT32, 0);
    if (!arr) return NULL;

    int32_t *data = (int32_t *)PyArray_DATA(arr);

    // Fill from per-sequence maps
    for (Py_ssize_t si = 0; si < n_seqs; ++si) {
        int32_t *row = data + si * n_cols;
        for (auto &kv : per_seq_maps[si]) {
            auto it = name_to_col.find(kv.first);
            if (it != name_to_col.end()) {
                row[it->second] = kv.second;
            }
        }
    }

    // Build Python list of k-mer names
    PyObject *name_list = PyList_New(n_cols);
    if (!name_list) { Py_DECREF(arr); return NULL; }
    for (int i = 0; i < n_cols; ++i) {
        PyObject *s = PyUnicode_FromStringAndSize(
            ordered_names[i].c_str(), (Py_ssize_t)ordered_names[i].size());
        if (!s) { Py_DECREF(arr); Py_DECREF(name_list); return NULL; }
        PyList_SET_ITEM(name_list, i, s);
    }

    PyObject *result = PyTuple_Pack(2, (PyObject *)arr, name_list);
    Py_DECREF(arr);
    Py_DECREF(name_list);
    return result;
}
