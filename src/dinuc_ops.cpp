// dinuc_ops.cpp -- Fast dinucleotide counting kernel.
//
// Provides pyprego_calc_sequences_dinucs: counts all 16 dinucleotides
// per sequence with OpenMP parallelism.

#include "_pyprego.h"
#include "dna_utils.h"

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// pyprego_calc_sequences_dinucs
// ---------------------------------------------------------------------------
// Python signature:
//   calc_sequences_dinucs(sequences: list[str]) -> ndarray[int64, (N, 16)]
//
// Counts occurrences of each of the 16 dinucleotides (AA, AC, AG, AT, CA, ...)
// in each sequence. Column order is base-4: first_base * 4 + second_base
// where A=0, C=1, G=2, T=3.

PyObject *pyprego_calc_sequences_dinucs(PyObject * /*self*/, PyObject *args)
{
    PyObject *seq_list;

    if (!PyArg_ParseTuple(args, "O", &seq_list))
        return NULL;

    if (!PyList_Check(seq_list)) {
        PyErr_SetString(PyExc_TypeError, "sequences must be a list of strings");
        return NULL;
    }

    const Py_ssize_t n_seqs = PyList_GET_SIZE(seq_list);

    // Extract sequences as std::string (uppercased)
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

        if (len < 2) {
            PyErr_Format(PyExc_ValueError,
                         "Sequence %zd is too short for dinucleotide counting (length %zd)",
                         i, len);
            return NULL;
        }

        seqs[i].assign(data, (size_t)len);
        // uppercase
        for (auto &ch : seqs[i]) {
            if (ch >= 'a' && ch <= 'z') ch -= 32;
        }
    }

    // Allocate output array (n_seqs, 16) of int64
    npy_intp dims[2] = {(npy_intp)n_seqs, 16};
    PyArrayObject *arr = (PyArrayObject *)PyArray_ZEROS(2, dims, NPY_INT64, 0);
    if (!arr) return NULL;

    int64_t *data = (int64_t *)PyArray_DATA(arr);

    // Parallel over sequences
    #pragma omp parallel for schedule(dynamic, 64)
    for (Py_ssize_t si = 0; si < n_seqs; ++si) {
        const std::string &seq = seqs[si];
        int slen = (int)seq.size();
        int64_t *row = data + si * 16;
        const char *s = seq.c_str();

        int prev = encode_char(s[0]);
        for (int j = 1; j < slen; ++j) {
            int curr = encode_char(s[j]);
            if (prev >= 0 && curr >= 0) {
                row[prev * 4 + curr]++;
            }
            prev = curr;
        }
    }

    return (PyObject *)arr;
}
