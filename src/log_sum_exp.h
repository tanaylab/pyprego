#ifndef _PYPREGO_LOG_SUM_EXP_H_
#define _PYPREGO_LOG_SUM_EXP_H_

// Header-only numerically stable log-space arithmetic.
// Ported from prego/src/logSumExp.h

#include <cmath>
#include <limits>

// In-place log(exp(l1) + exp(l2)), result stored in l1.
inline void log_sum_log(double &l1, double l2)
{
    if (l1 > l2) {
        if (!std::isinf(l2)) {
            l1 += std::log(1.0 + std::exp(l2 - l1));
        }
    } else {
        if (std::isinf(l1)) {
            l1 = l2;
        } else {
            l1 = l2 + std::log(1.0 + std::exp(l1 - l2));
        }
    }
}

// float overload
inline void log_sum_log(float &l1, float l2)
{
    if (l1 > l2) {
        if (!std::isinf(l2)) {
            l1 += std::log(1.0f + std::exp(l2 - l1));
        }
    } else {
        if (std::isinf(l1)) {
            l1 = l2;
        } else {
            l1 = l2 + std::log(1.0f + std::exp(l1 - l2));
        }
    }
}

// Numerically stable log(sum(exp(x[0..n-1]))).
// Uses the max-shift trick to avoid overflow/underflow.
inline double log_sum_exp(const double *x, int n)
{
    if (n == 0) return -std::numeric_limits<double>::infinity();
    if (n == 1) return x[0];

    // Find maximum
    double max_val = x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] > max_val) max_val = x[i];
    }

    if (std::isinf(max_val)) return max_val;

    // Compute sum of exp(x[i] - max_val)
    double sum = 0.0;
    constexpr int BLOCK_SIZE = 4;

    int i = 0;
    for (; i <= n - BLOCK_SIZE; i += BLOCK_SIZE) {
        sum += std::exp(x[i]     - max_val);
        sum += std::exp(x[i + 1] - max_val);
        sum += std::exp(x[i + 2] - max_val);
        sum += std::exp(x[i + 3] - max_val);
    }
    for (; i < n; ++i) {
        sum += std::exp(x[i] - max_val);
    }

    return max_val + std::log(sum);
}

// float overload
inline float log_sum_exp(const float *x, int n)
{
    if (n == 0) return -std::numeric_limits<float>::infinity();
    if (n == 1) return x[0];

    float max_val = x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] > max_val) max_val = x[i];
    }

    if (std::isinf(max_val)) return max_val;

    float sum = 0.0f;
    constexpr int BLOCK_SIZE = 4;

    int i = 0;
    for (; i <= n - BLOCK_SIZE; i += BLOCK_SIZE) {
        sum += std::exp(x[i]     - max_val);
        sum += std::exp(x[i + 1] - max_val);
        sum += std::exp(x[i + 2] - max_val);
        sum += std::exp(x[i + 3] - max_val);
    }
    for (; i < n; ++i) {
        sum += std::exp(x[i] - max_val);
    }

    return max_val + std::log(sum);
}

#endif // _PYPREGO_LOG_SUM_EXP_H_
