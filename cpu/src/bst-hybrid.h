/*
 * Copyright 2018-2020 Kyle Berney
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 *    http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef BST_HYBRID_H
#define BST_HYBRID_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#include "common.h"
#include "cycles.h"
#include "involutions.h"

//Permutes sorted array into BST layout for n = 2^d - 1
//Involution approach
template<typename TYPE>
void permute(TYPE *A, uint64_t n, uint32_t d) {
    uint64_t j;
    TYPE temp;

    //Phase 1
    for (uint64_t i = 0; i < n; ++i) {
        j = rev_d(i+1, d) - 1;

        if (i < j) {
            temp = A[i];
            A[i] = A[j];
            A[j] = temp;
        }
    }

    //Phase 2
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
}

//Permutes sorted array into BST layout for n = 2^d - 1 using p processors
//Involution approach
template<typename TYPE>
void permute_parallel(TYPE *A, uint64_t n, uint32_t d, uint32_t p) {
    uint64_t j;
    TYPE temp;

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
}

//Gathers and shifts non-full level of leaves to the end of the array
//Cycle-leader approach
template<typename TYPE>
void permute_leaves(TYPE *A, uint64_t n, uint64_t numInternals, uint64_t numLeaves) {
    extended_equidistant_gather2<TYPE>(A, 2*numLeaves, 1);
    shift_right<TYPE>(&A[numLeaves], numInternals, numInternals - numLeaves);
}

//Gathers and shifts non-full level of leaves to the end of the array using p processors
//Cycle-leader approach
template<typename TYPE>
void permute_leaves_parallel(TYPE *A, uint64_t n, uint64_t numInternals, uint64_t numLeaves, uint32_t p) {
    extended_equidistant_gather2_parallel<TYPE>(A, 2*numLeaves, 1, p);
    shift_right_parallel<TYPE>(&A[numLeaves], numInternals, numInternals - numLeaves, p);
}

template<typename TYPE>
double timePermuteBST(TYPE *A, uint64_t n, uint32_t p) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    uint32_t h = log2(n);
    if (n != pow(2, h+1) - 1) {     //non-full tree
        uint64_t numInternals = pow(2, h) - 1;
        uint64_t numLeaves = n - numInternals;
        if (p == 1) {
            permute_leaves<TYPE>(A, n, numInternals, numLeaves);
            permute<TYPE>(A, n - numLeaves, h);
        }
        else {
            permute_leaves_parallel<TYPE>(A, n, numInternals, numLeaves, p);
            permute_parallel<TYPE>(A, n - numLeaves, h, p);
        }
    }
    else {    //full tree
        if (p == 1) permute<TYPE>(A, n, h+1);
        else permute_parallel<TYPE>(A, n, h+1, p);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double ms = ((end.tv_sec*1000000000. + end.tv_nsec) - (start.tv_sec*1000000000. + start.tv_nsec)) / 1000000.;   //millisecond
    return ms;
}
#endif