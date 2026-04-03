#ifndef _PYPREGO_DNA_UTILS_H_
#define _PYPREGO_DNA_UTILS_H_

// Header-only DNA encoding utilities for pyprego C extension.

#include <cstdint>

// Encode an ASCII nucleotide character to an integer index.
// A/a -> 0, C/c -> 1, G/g -> 2, T/t -> 3, anything else -> -1
inline int encode_char(char c)
{
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default:            return -1;
    }
}

// Return the complement index: A(0)<->T(3), C(1)<->G(2)
inline int complement_idx(int idx)
{
    return 3 - idx;
}

// Check whether an encoded base index is valid (0-3).
inline bool is_valid_base(int idx)
{
    return idx >= 0 && idx <= 3;
}

#endif // _PYPREGO_DNA_UTILS_H_
