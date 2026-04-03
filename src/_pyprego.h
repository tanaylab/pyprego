#ifndef _PYPREGO_H_
#define _PYPREGO_H_

// This must be defined before "#include <numpy/arrayobject.h>" in all files
// that use numpy EXCEPT the init file (which #undef's NO_IMPORT_ARRAY).
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL _pyprego_ARRAY_API

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

// ---------------------------------------------------------------------------
// Array element access helpers (1-D contiguous arrays)
// ---------------------------------------------------------------------------

inline double  PPDOUBLE(PyArrayObject *arr, npy_intp i) { return *(double *)PyArray_GETPTR1(arr, i); }
inline long    PPLONG(PyArrayObject *arr, npy_intp i)   { return *(long *)PyArray_GETPTR1(arr, i); }
inline int     PPINT(PyArrayObject *arr, npy_intp i)    { return *(int *)PyArray_GETPTR1(arr, i); }
inline bool    PPBOOL(PyArrayObject *arr, npy_intp i)   { return *(bool *)PyArray_GETPTR1(arr, i); }

inline double &PPDOUBLE_REF(PyArrayObject *arr, npy_intp i) { return *(double *)PyArray_GETPTR1(arr, i); }
inline long   &PPLONG_REF(PyArrayObject *arr, npy_intp i)   { return *(long *)PyArray_GETPTR1(arr, i); }
inline int    &PPINT_REF(PyArrayObject *arr, npy_intp i)    { return *(int *)PyArray_GETPTR1(arr, i); }

// 2-D element access
inline double  PPDOUBLE2(PyArrayObject *arr, npy_intp i, npy_intp j) { return *(double *)PyArray_GETPTR2(arr, i, j); }
inline double &PPDOUBLE2_REF(PyArrayObject *arr, npy_intp i, npy_intp j) { return *(double *)PyArray_GETPTR2(arr, i, j); }

// Dimension helpers
inline bool    PPIS1D(PyArrayObject *arr) { return arr && PyArray_Check(arr) && PyArray_NDIM(arr) == 1; }
inline bool    PPIS2D(PyArrayObject *arr) { return arr && PyArray_Check(arr) && PyArray_NDIM(arr) == 2; }
inline npy_intp PPLEN(PyArrayObject *arr) { return PyArray_DIM(arr, 0); }
inline npy_intp PPNCOL(PyArrayObject *arr) { return PyArray_DIM(arr, 1); }

// Contiguous data pointer (caller must ensure correct dtype)
inline double *PPDATA_DOUBLE(PyArrayObject *arr) { return (double *)PyArray_DATA(arr); }
inline long   *PPDATA_LONG(PyArrayObject *arr)   { return (long *)PyArray_DATA(arr); }
inline int    *PPDATA_INT(PyArrayObject *arr)     { return (int *)PyArray_DATA(arr); }

// ---------------------------------------------------------------------------
// Global exception object (defined in _pyprego_init.cpp)
// ---------------------------------------------------------------------------
extern PyObject *s_pyprego_err;

#endif // _PYPREGO_H_
