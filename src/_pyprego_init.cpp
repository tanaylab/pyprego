// _pyprego_init.cpp -- Module initialization for the _pyprego C extension.
// This is the ONLY file that imports the numpy C-API (import_array).

// Must undef NO_IMPORT_ARRAY before including numpy so that
// this translation unit actually defines the API table.
#ifdef NO_IMPORT_ARRAY
    #undef NO_IMPORT_ARRAY
#endif

#ifndef PY_ARRAY_UNIQUE_SYMBOL
    #define PY_ARRAY_UNIQUE_SYMBOL _pyprego_ARRAY_API
#endif

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------
PyObject *g_pyprego_module = nullptr;
PyObject *s_pyprego_err    = nullptr;

// ---------------------------------------------------------------------------
// Forward declarations of module-level functions
// ---------------------------------------------------------------------------

// Stub: returns the extension version string.
static PyObject *pyprego_version(PyObject *self, PyObject *args)
{
    return PyUnicode_FromString("0.0.1");
}

// Defined in kmer_ops.cpp
extern PyObject *pyprego_kmer_matrix(PyObject *self, PyObject *args);

// Defined in energy_ops.cpp
extern PyObject *pyprego_init_energies(PyObject *self, PyObject *args);
extern PyObject *pyprego_batch_extract_energies(PyObject *self, PyObject *args);

// Defined in score_ops.cpp
extern PyObject *pyprego_choose_best_move(PyObject *self, PyObject *args);

// Defined in dinuc_ops.cpp
extern PyObject *pyprego_calc_sequences_dinucs(PyObject *self, PyObject *args);

// Defined in screen_kmer_ops.cpp
extern PyObject *pyprego_screen_kmers(PyObject *self, PyObject *args);

// ---------------------------------------------------------------------------
// Method table
// ---------------------------------------------------------------------------
static PyMethodDef module_methods[] = {
    {"version", pyprego_version, METH_NOARGS, "Return the _pyprego extension version string."},
    {"kmer_matrix", pyprego_kmer_matrix, METH_VARARGS,
     "kmer_matrix(sequences, k, max_gap=0) -> (ndarray[int32], list[str])\n"
     "Count k-mer occurrences in each sequence (C++ with OpenMP)."},
    {"init_energies", pyprego_init_energies, METH_VARARGS,
     "init_energies(encoded, nuc_factors, spat_factors, train_mask, "
     "spat_bin_size, bidirect, symmetrize_spat, derivs, spat_derivs) -> None\n"
     "Compute PSSM energy derivatives for all sequences (C++ with OpenMP)."},
    {"batch_extract_energies", pyprego_batch_extract_energies, METH_VARARGS,
     "batch_extract_energies(encoded, log_pssm_list, spat_factors_list, "
     "spat_bin_sizes, bidirect, output) -> None\n"
     "Batch compute PWM energies for multiple motifs (C++ with OpenMP)."},
    {"choose_best_move", pyprego_choose_best_move, METH_VARARGS,
     "choose_best_move(derivs, nuc_factors, response, ...) -> (best_pos, best_step, scores)\n"
     "Evaluate all candidate moves and return the best one (C++)."},
    {"calc_sequences_dinucs", pyprego_calc_sequences_dinucs, METH_VARARGS,
     "calc_sequences_dinucs(sequences: list[str]) -> ndarray[int64, (N, 16)]\n"
     "Count dinucleotide occurrences in each sequence (C++ with OpenMP)."},
    {"screen_kmers", pyprego_screen_kmers, METH_VARARGS,
     "screen_kmers(sequences, response, kmer_length, min_cor, min_gap=0, max_gap=0) -> tuple\n"
     "Screen k-mers (pure and gapped) for correlation with response.\n"
     "Returns (names, max_r2, correlations, avg_n, avg_var)."},
    {NULL, NULL, 0, NULL}
};

// ---------------------------------------------------------------------------
// Module cleanup
// ---------------------------------------------------------------------------
static void pyprego_module_free(void * /*m*/)
{
    s_pyprego_err = nullptr;
}

// ---------------------------------------------------------------------------
// Module init
// ---------------------------------------------------------------------------
PyMODINIT_FUNC PyInit__pyprego(void)
{
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_pyprego",
        "pyprego C++ extension -- fast kernels for PWM regression",
        -1,
        module_methods,
        NULL,
        NULL,
        NULL,
        pyprego_module_free
    };

    g_pyprego_module = PyModule_Create(&moduledef);
    if (!g_pyprego_module)
        return NULL;

    // Create a custom exception: pyprego.error
    s_pyprego_err = PyErr_NewException("pyprego.error", NULL, NULL);
    if (!s_pyprego_err) {
        Py_DECREF(g_pyprego_module);
        return NULL;
    }
    if (PyModule_AddObject(g_pyprego_module, "error", s_pyprego_err) < 0) {
        Py_DECREF(s_pyprego_err);
        s_pyprego_err = NULL;
        Py_DECREF(g_pyprego_module);
        return NULL;
    }

    // Import numpy C-API
    import_array();

    return g_pyprego_module;
}
