#ifndef BTREE_INVOLUTIONS_H
#define BTREE_INVOLUTIONS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#include "common.h"
#include "involutions.h"

//permutes sorted array into level order btree layout for n = (b + 1)^d - 1
template<typename TYPE>
void permute(TYPE *A, uint64_t n, uint64_t b, uint64_t d) {
    while (d > 1) {
        unshuffle<TYPE>(A, b+1, n, d);
        shuffle_dk<TYPE>(&A[n/(b+1)], b, pow(b+1, d-1) * b);
        n /= b+1;
        d--;
    }
}

//permutes sorted array into level order btree layout for n = (b + 1)^d - 1 using p threads
template<typename TYPE>
void permute_parallel(TYPE *A, uint64_t n, uint64_t b, uint64_t d, uint32_t p) {
    while (d > 1) {
        unshuffle_parallel<TYPE>(A, b+1, n, d, p);
        shuffle_dk_parallel<TYPE>(&A[n/(b+1)], b, pow(b+1, d-1) * b, p);
        n /= b+1;
        d--;
    }
}

//Gathers and shifts non-full level of leaves to the end of the array
template<typename TYPE>
void permute_leaves(TYPE *A, uint64_t n, uint64_t b, uint64_t numInternals, uint64_t numLeaves) {
    uint64_t r = numLeaves % b; 	//number of leaves belonging to a non-full node
    uint64_t l = numLeaves - r;		//number of leaves belonging to full nodes
    uint64_t i = l/b;       	    //number of internals partitioning the leaf nodes
    
    unshuffle_dk<TYPE>(A, b+1, l + i);
    shift_right<TYPE>(A, l + i, i);
    shuffle_dk<TYPE>(&A[i], b, l);
    shift_right<TYPE>(&A[i], numLeaves + numInternals - i, numInternals - i);
}

//Gathers and shifts non-full level of leaves to the end of the array using p threads
template<typename TYPE>
void permute_leaves_parallel(TYPE *A, uint64_t n, uint64_t b, uint64_t numInternals, uint64_t numLeaves, uint32_t p) {
    uint64_t r = numLeaves % b;     //number of leaves belonging to a non-full node
    uint64_t l = numLeaves - r;     //number of leaves belonging to full nodes
    uint64_t i = l/b;               //number of internals partitioning the leaf nodes
    
    unshuffle_dk_parallel<TYPE>(A, b+1, l + i, p);
    shift_right_parallel<TYPE>(A, l + i, i, p);
    shuffle_dk_parallel<TYPE>(&A[i], b, l, p);
    shift_right_parallel<TYPE>(&A[i], numLeaves + numInternals - i, numInternals - i, p);
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
            permute<TYPE>(A, n - numLeaves, b, h);
        }
        else {
            permute_leaves_parallel<TYPE>(A, n, b, numInternals, numLeaves, p);
            permute_parallel<TYPE>(A, n - numLeaves, b, h, p);
        }
    }
    else {    //full tree
        if (p == 1) permute<TYPE>(A, n, b, h+1);
        else permute_parallel<TYPE>(A, n, b, h+1, p);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double ms = ((end.tv_sec*1000000000. + end.tv_nsec) - (start.tv_sec*1000000000. + start.tv_nsec)) / 1000000.;   //millisecond
    return ms;
}
#endif