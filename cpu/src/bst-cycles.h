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

#ifndef BST_CYCLES_H
#define BST_CYCLES_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#include "common.h"
#include "cycles.h"

//Permutes sorted array into BST layout for n = 2^d - 1, for some integer d
template<typename TYPE>
void permute(TYPE *A, uint64_t n) {
    if (n == 1) return;

    extended_equidistant_gather<TYPE>(A, n, 1);
    permute<TYPE>(A, n/2);
}

//Permutes sorted array into BST layout for n = 2^d - 1, for some integer d using p processors
template<typename TYPE>
void permute_parallel(TYPE *A, uint64_t n, uint32_t p) {
    if (n == 1) return;

    extended_equidistant_gather_parallel<TYPE>(A, n, 1, p);
    permute_parallel<TYPE>(A, n/2, p);
}

template<typename TYPE>
double timePermuteBST(TYPE *A, uint64_t n, uint32_t p) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    uint64_t h = log2(n);
    if (n != pow(2, h+1) - 1) {		//non-full tree
        /*uint64_t numInternals = pow(2, h) - 1;
        uint64_t numLeaves = n - numInternals;
        permute_leaves<TYPE>(A, n, b, numInternals, numLeaves);
        permute<TYPE>(A, n - numLeaves, b);*/
        printf("Non-perfect BST ==> NOT YET IMPLEMENTED!\n");
    }
    else {    //full tree
        if (p == 1) permute<TYPE>(A, n);
        else permute_parallel<TYPE>(A, n, p);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double ms = ((end.tv_sec*1000000000. + end.tv_nsec) - (start.tv_sec*1000000000. + start.tv_nsec)) / 1000000.;   //millisecond
    return ms;
}
#endif