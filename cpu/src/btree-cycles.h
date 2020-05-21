#ifndef BTREE_CYCLES_H
#define BTREE_CYCLES_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#include "common.h"
#include "cycles.h"

//Gathers and shifts non-full level of leaves to the end of the array
template<typename TYPE>
void permute_leaves(TYPE *A, uint64_t n, uint64_t b, uint64_t numInternals, uint64_t numLeaves) {
    uint64_t r = numLeaves % b; 	//number of leaves belonging to a non-full node
    uint64_t l = numLeaves - r;		//number of leaves belonging to full nodes
    uint64_t i = l/b;       		//number of internals partitioning the leaf nodes

    extended_equidistant_gather2<TYPE>(A, l + i, b);
    shift_right<TYPE>(&A[i], numLeaves + numInternals - i, numInternals - i);
}

//Gathers and shifts non-full level of leaves to the end of the array using p processors
template<typename TYPE>
void permute_leaves_parallel(TYPE *A, uint64_t n, uint64_t b, uint64_t numInternals, uint64_t numLeaves, uint32_t p) {
    uint64_t r = numLeaves % b;     //number of leaves belonging to a non-full node
    uint64_t l = numLeaves - r;     //number of leaves belonging to full nodes
    uint64_t i = l/b;               //number of internals partitioning the leaf nodes

    extended_equidistant_gather2_parallel<TYPE>(A, l + i, b, p);
    shift_right_parallel<TYPE>(&A[i], numLeaves + numInternals - i, numInternals - i, p);
}

//Permutes sorted array into level order btree layout for n = (b + 1)^d - 1, for some integer d
template<typename TYPE>
void permute(TYPE *A, uint64_t n, uint64_t b) {
    if (n == b) return;

    extended_equidistant_gather<TYPE>(A, n, b);
    permute<TYPE>(A, n/(b+1), b);
}

//Permutes sorted array into level order btree layout for n = (b + 1)^d - 1, for some integer d using p processors
template<typename TYPE>
void permute_parallel(TYPE *A, uint64_t n, uint64_t b, uint32_t p) {
    if (n == b) return;

    extended_equidistant_gather_parallel<TYPE>(A, n, b, p);
    permute_parallel<TYPE>(A, n/(b+1), b, p);
}

template<typename TYPE>
double timePermuteBtree(TYPE *A, uint64_t n, uint64_t b, uint32_t p) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    uint64_t h = log10(n)/log10(b+1);
    if (n != pow(b+1, h+1) - 1) {		//non-full tree
        uint64_t numInternals = pow(b+1, h) - 1;
        uint64_t numLeaves = n - numInternals;
        if (p == 1) {
            permute_leaves<TYPE>(A, n, b, numInternals, numLeaves);
            permute<TYPE>(A, n - numLeaves, b);
        }
        else {
            permute_leaves_parallel<TYPE>(A, n, b, numInternals, numLeaves, p);
            permute_parallel<TYPE>(A, n - numLeaves, b, p);
        }
    }
    else {    //full tree
        if (p == 1) permute<TYPE>(A, n, b);
        else permute_parallel<TYPE>(A, n, b, p);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double ms = ((end.tv_sec*1000000000. + end.tv_nsec) - (start.tv_sec*1000000000. + start.tv_nsec)) / 1000000.;   //millisecond
    return ms;
}
#endif