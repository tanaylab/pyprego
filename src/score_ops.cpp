// score_ops.cpp -- Regression scoring and move selection kernels.
//
// Provides pyprego_choose_best_move: a C++ implementation of
// _PWMLRegression.choose_best_move() with R2 and KS scoring modes.

#include "_pyprego.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// pyprego_choose_best_move
// ---------------------------------------------------------------------------
// Python signature:
//   choose_best_move(derivs, nuc_factors, response, train_mask, folds,
//                    fold_sizes, data_avg_fold, data_var_fold,
//                    data_epsilon, neigh_nuc_indices, neigh_deltas,
//                    neigh_sizes, min_prob, motif_len, num_folds, rdim,
//                    log_energy, energy_epsilon, score_metric)
//                    -> (best_pos, best_step, scores_array)
//
// Parameters:
//   derivs           : ndarray[float64, (N, K, 4)]  - derivatives from init_energies
//   nuc_factors      : ndarray[float64, (K, 4)]     - current PSSM
//   response         : ndarray[float64, (N, rdim)]   - response values
//   train_mask       : ndarray[bool, (N,)]           - training mask
//   folds            : ndarray[int32, (N,)]          - fold assignments
//   fold_sizes       : ndarray[int32, (num_folds,)]  - samples per fold
//   data_avg_fold    : ndarray[float64, (num_folds, rdim)] - per-fold response means
//   data_var_fold    : ndarray[float64, (num_folds, rdim)] - per-fold response variances
//   data_epsilon     : ndarray[float64, (N,)]        - random epsilon for tie-breaking (KS only)
//   neigh_nuc_indices: ndarray[int32, (n_neigh, 2)]  - nucleotide indices for each move
//   neigh_deltas     : ndarray[float64, (n_neigh, 2)] - delta values for each move
//   neigh_sizes      : ndarray[int32, (n_neigh,)]    - number of (nuc, delta) pairs per move
//   min_prob         : float                         - minimum nucleotide probability
//   motif_len        : int                           - K
//   num_folds        : int                           - number of CV folds
//   rdim             : int                           - response dimensionality
//   log_energy       : int                           - whether to apply log transform
//   energy_epsilon   : float                         - epsilon for log(energy + eps)
//   score_metric     : str                           - "r2" or "ks"
//
// Returns: tuple(best_pos: int, best_step: int, fold_scores: ndarray[float64, (num_folds, n_moves)])

PyObject *pyprego_choose_best_move(PyObject * /*self*/, PyObject *args)
{
    PyArrayObject *py_derivs = nullptr;
    PyArrayObject *py_nuc_factors = nullptr;
    PyArrayObject *py_response = nullptr;
    PyArrayObject *py_train_mask = nullptr;
    PyArrayObject *py_folds = nullptr;
    PyArrayObject *py_fold_sizes = nullptr;
    PyArrayObject *py_data_avg_fold = nullptr;
    PyArrayObject *py_data_var_fold = nullptr;
    PyArrayObject *py_data_epsilon = nullptr;
    PyArrayObject *py_neigh_nuc_indices = nullptr;
    PyArrayObject *py_neigh_deltas = nullptr;
    PyArrayObject *py_neigh_sizes = nullptr;
    double min_prob = 0.0;
    int motif_len = 0;
    int num_folds = 0;
    int rdim = 0;
    int log_energy = 0;
    double energy_epsilon = 0.0;
    const char *score_metric = nullptr;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!O!O!O!O!diiiids",
                          &PyArray_Type, &py_derivs,
                          &PyArray_Type, &py_nuc_factors,
                          &PyArray_Type, &py_response,
                          &PyArray_Type, &py_train_mask,
                          &PyArray_Type, &py_folds,
                          &PyArray_Type, &py_fold_sizes,
                          &PyArray_Type, &py_data_avg_fold,
                          &PyArray_Type, &py_data_var_fold,
                          &PyArray_Type, &py_data_epsilon,
                          &PyArray_Type, &py_neigh_nuc_indices,
                          &PyArray_Type, &py_neigh_deltas,
                          &PyArray_Type, &py_neigh_sizes,
                          &min_prob,
                          &motif_len,
                          &num_folds,
                          &rdim,
                          &log_energy,
                          &energy_epsilon,
                          &score_metric))
        return NULL;

    // Get dimensions
    const npy_intp N = PyArray_DIM(py_derivs, 0);
    const npy_intp K = motif_len;
    const int n_neigh = (int)PyArray_DIM(py_neigh_sizes, 0);
    const int n_moves = (int)(K * n_neigh);

    // Get data pointers - we require C-contiguous arrays
    const double *derivs = (const double *)PyArray_DATA(py_derivs);
    const double *nuc_factors = (const double *)PyArray_DATA(py_nuc_factors);
    const double *response = (const double *)PyArray_DATA(py_response);
    const npy_bool *train_mask = (const npy_bool *)PyArray_DATA(py_train_mask);
    const int *folds = (const int *)PyArray_DATA(py_folds);
    const int *fold_sizes = (const int *)PyArray_DATA(py_fold_sizes);
    const double *data_avg_fold = (const double *)PyArray_DATA(py_data_avg_fold);
    const double *data_var_fold = (const double *)PyArray_DATA(py_data_var_fold);
    const double *data_epsilon = (const double *)PyArray_DATA(py_data_epsilon);
    const int *neigh_nuc_indices = (const int *)PyArray_DATA(py_neigh_nuc_indices);
    const double *neigh_deltas = (const double *)PyArray_DATA(py_neigh_deltas);
    const int *neigh_sizes = (const int *)PyArray_DATA(py_neigh_sizes);

    const bool is_r2 = (strcmp(score_metric, "r2") == 0);

    // ── Pre-compute all candidate probability vectors: (n_moves, 4) ──
    std::vector<double> all_probs(n_moves * 4);

    for (int pos = 0; pos < K; ++pos) {
        for (int step = 0; step < n_neigh; ++step) {
            int move_idx = pos * n_neigh + step;
            double *probs = &all_probs[move_idx * 4];
            // Copy current PSSM row
            for (int nuc = 0; nuc < 4; ++nuc) {
                probs[nuc] = nuc_factors[pos * 4 + nuc];
            }
            // Apply neighbourhood perturbation
            int nsz = neigh_sizes[step];
            for (int ni = 0; ni < nsz; ++ni) {
                int nuc_idx = neigh_nuc_indices[step * 2 + ni];
                double delta = neigh_deltas[step * 2 + ni];
                probs[nuc_idx] += delta;
                if (probs[nuc_idx] <= 0.0) {
                    probs[nuc_idx] = min_prob;
                }
            }
        }
    }

    // Allocate output scores array: (num_folds, n_moves)
    npy_intp score_dims[2] = {(npy_intp)num_folds, (npy_intp)n_moves};
    PyArrayObject *py_scores = (PyArrayObject *)PyArray_ZEROS(2, score_dims, NPY_FLOAT64, 0);
    if (!py_scores) return NULL;
    double *scores = (double *)PyArray_DATA(py_scores);

    if (is_r2) {
        // ── R2 scoring ──
        // For each position, batch-compute energies for all moves at that position
        for (int pos = 0; pos < K; ++pos) {
            int move_start = pos * n_neigh;

            // probs_at_pos: (n_neigh, 4) starting at all_probs[move_start * 4]
            // derivs_at_pos for seq s: derivs[s * K * 4 + pos * 4 + nuc]

            // Compute energies for all sequences and all moves at this position
            // energies[s, m] = sum_nuc derivs[s, pos, nuc] * probs[move_start + m, nuc]
            // Shape: (N, n_neigh)
            std::vector<double> energies(N * n_neigh, 0.0);

            #pragma omp parallel for schedule(static)
            for (npy_intp s = 0; s < N; ++s) {
                if (!train_mask[s]) continue;
                const double *d = &derivs[s * K * 4 + pos * 4];
                for (int m = 0; m < n_neigh; ++m) {
                    const double *p = &all_probs[(move_start + m) * 4];
                    double e = d[0] * p[0] + d[1] * p[1] + d[2] * p[2] + d[3] * p[3];
                    energies[s * n_neigh + m] = e;
                }
            }

            // Apply log transform if needed
            if (log_energy) {
                #pragma omp parallel for schedule(static)
                for (npy_intp s = 0; s < N; ++s) {
                    if (!train_mask[s]) continue;
                    for (int m = 0; m < n_neigh; ++m) {
                        energies[s * n_neigh + m] = std::log(energies[s * n_neigh + m] + energy_epsilon);
                    }
                }
            }

            // For each fold, compute R2 for all moves
            for (int fi = 0; fi < num_folds; ++fi) {
                int fold_n = fold_sizes[fi];
                if (fold_n == 0) continue;
                double inv_fold_n = 1.0 / (double)fold_n;

                // Sum of energies and sum of squared energies for each move in this fold
                std::vector<double> sum_e(n_neigh, 0.0);
                std::vector<double> sum_e2(n_neigh, 0.0);

                // Sum of energy * response for each move and response dimension
                std::vector<double> sum_er(n_neigh * rdim, 0.0);

                for (npy_intp s = 0; s < N; ++s) {
                    if (!train_mask[s]) continue;
                    if (num_folds > 1 && folds[s] != fi) continue;

                    for (int m = 0; m < n_neigh; ++m) {
                        double e = energies[s * n_neigh + m];
                        sum_e[m] += e;
                        sum_e2[m] += e * e;
                        for (int rd = 0; rd < rdim; ++rd) {
                            sum_er[m * rdim + rd] += e * response[s * rdim + rd];
                        }
                    }
                }

                // Compute R2 for each move
                for (int m = 0; m < n_neigh; ++m) {
                    double ex = sum_e[m] * inv_fold_n;
                    double ex2 = sum_e2[m] * inv_fold_n;
                    double pred_var = ex2 - ex * ex;

                    double tot_r2 = 0.0;
                    for (int rd = 0; rd < rdim; ++rd) {
                        double xy = sum_er[m * rdim + rd] * inv_fold_n;
                        double fold_avg = data_avg_fold[fi * rdim + rd];
                        double fold_var = data_var_fold[fi * rdim + rd];
                        double cov = xy - ex * fold_avg;
                        double denom = pred_var * fold_var;
                        if (denom > 0.0) {
                            tot_r2 += cov * cov / denom;
                        }
                    }

                    scores[fi * n_moves + move_start + m] = tot_r2;
                }
            }
        }
    } else {
        // ── KS scoring ──
        for (int pos = 0; pos < K; ++pos) {
            int move_start = pos * n_neigh;

            // Compute energies for all sequences and all moves at this position
            std::vector<double> energies(N * n_neigh, 0.0);

            #pragma omp parallel for schedule(static)
            for (npy_intp s = 0; s < N; ++s) {
                if (!train_mask[s]) continue;
                const double *d = &derivs[s * K * 4 + pos * 4];
                for (int m = 0; m < n_neigh; ++m) {
                    const double *p = &all_probs[(move_start + m) * 4];
                    double e = d[0] * p[0] + d[1] * p[1] + d[2] * p[2] + d[3] * p[3];
                    energies[s * n_neigh + m] = e;
                }
            }

            // For each fold and each move, compute KS
            for (int fi = 0; fi < num_folds; ++fi) {
                // Collect fold indices
                std::vector<int> fold_idx;
                fold_idx.reserve(fold_sizes[fi]);
                for (npy_intp s = 0; s < N; ++s) {
                    if (!train_mask[s]) continue;
                    if (num_folds > 1 && folds[s] != fi) continue;
                    fold_idx.push_back((int)s);
                }
                int fold_n = (int)fold_idx.size();
                if (fold_n == 0) continue;

                // response for fold sequences (column 0 for KS)
                std::vector<double> resp_fold(fold_n);
                std::vector<double> eps_fold(fold_n);
                for (int i = 0; i < fold_n; ++i) {
                    resp_fold[i] = response[fold_idx[i] * rdim];
                    eps_fold[i] = data_epsilon[fold_idx[i]];
                }

                // Count n_0 and n_1
                double n_1 = 0.0, n_0 = 0.0;
                for (int i = 0; i < fold_n; ++i) {
                    if (resp_fold[i] == 0.0) n_0 += 1.0;
                    else n_1 += 1.0;
                }
                if (n_0 == 0.0 || n_1 == 0.0) continue;

                double inv_n0 = 1.0 / n_0;
                double inv_n1 = 1.0 / n_1;

                // Precompute step increments
                std::vector<double> steps(fold_n);
                for (int i = 0; i < fold_n; ++i) {
                    steps[i] = (resp_fold[i] == 0.0) ? -inv_n0 : inv_n1;
                }

                // For each move
                std::vector<double> sort_keys(fold_n);
                std::vector<int> order(fold_n);

                for (int m = 0; m < n_neigh; ++m) {
                    // Compute sort keys: -(energy * (1 + eps))
                    for (int i = 0; i < fold_n; ++i) {
                        double e = energies[fold_idx[i] * n_neigh + m];
                        sort_keys[i] = -(e * (1.0 + eps_fold[i]));
                    }

                    // Argsort
                    std::iota(order.begin(), order.end(), 0);
                    std::sort(order.begin(), order.end(),
                              [&sort_keys](int a, int b) {
                                  return sort_keys[a] < sort_keys[b];
                              });

                    // Cumulative sum of steps in sorted order, track max
                    double max_diff = 0.0;
                    double cur_diff = 0.0;
                    for (int i = 0; i < fold_n; ++i) {
                        cur_diff += steps[order[i]];
                        if (cur_diff > max_diff) max_diff = cur_diff;
                    }

                    scores[fi * n_moves + move_start + m] = max_diff;
                }
            }
        }
    }

    // ── Rank within each fold and compute average ranks ──
    std::vector<int> avg_ranks(n_moves, 0);

    {
        std::vector<int> order(n_moves);
        std::vector<int> ranks(n_moves);

        for (int fi = 0; fi < num_folds; ++fi) {
            double *fold_scores = &scores[fi * n_moves];

            // Argsort: ascending order of scores
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(),
                      [fold_scores](int a, int b) {
                          return fold_scores[a] < fold_scores[b];
                      });

            // Assign ranks: rank 0 = worst, rank n_moves-1 = best
            for (int i = 0; i < n_moves; ++i) {
                ranks[order[i]] = i;
            }

            for (int i = 0; i < n_moves; ++i) {
                avg_ranks[i] += ranks[i];
            }
        }
    }

    // Integer division by num_folds (matching Python's // operator)
    for (int i = 0; i < n_moves; ++i) {
        avg_ranks[i] /= num_folds;
    }

    // Find best move (highest average rank)
    int best_idx = 0;
    for (int i = 1; i < n_moves; ++i) {
        if (avg_ranks[i] > avg_ranks[best_idx]) {
            best_idx = i;
        }
    }

    int best_pos = best_idx / n_neigh;
    int best_step = best_idx % n_neigh;

    // Return (best_pos, best_step, scores_array)
    PyObject *result = Py_BuildValue("(iiN)", best_pos, best_step, (PyObject *)py_scores);
    return result;
}
