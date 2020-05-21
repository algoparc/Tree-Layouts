#ifndef BST_INVOLUTIONS_H
#define BST_INVOLUTIONS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#include "common.h"
#include "involutions.h"

template<typename TYPE>
double timePermuteBST(TYPE *A, uint64_t n, uint64_t d, uint32_t p) {
    uint64_t j;
    TYPE temp;
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    //Phase 1
    #pragma omp parallel for shared(A, n, d) private(j, temp) schedule(guided, B) num_threads(p)
    for (uint64_t i = 0; i < n; ++i) {
        j = rev_d(i+1, d) - 1;

        if (i < j) {
            temp = A[i];
            A[i] = A[j];
            A[j] = temp;
        }
    }

    //Phase 2
    #pragma omp parallel for shared(A, n, d) private(j, temp) schedule(guided, B) num_threads(p)
    for (uint64_t i = 0; i < n; ++i) {
        if (i+1 > (n-1)/2) { //leafs
            j = rev_b(i+1, d, d-1) - 1;
        }
        else {  //internals
            j = rev_b(i+1, d, d - (__builtin_clzll(i+1) - (64 - d) + 1)) - 1;
        }

        if (i < j) {
            temp = A[i];
            A[i] = A[j];
            A[j] = temp;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double ms = ((end.tv_sec*1000000000. + end.tv_nsec) - (start.tv_sec*1000000000. + start.tv_nsec)) / 1000000.;   //millisecond
    return ms;
}
#endif