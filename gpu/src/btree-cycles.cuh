/*
 * Copyright 2018-2021 Kyle Berney, Ben Karsin
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

#ifndef BTREE_CYCLES_CUH
#define BTREE_CYCLES_CUH

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#include "common.cuh"
#include "cycles.cuh"

//Gathers and shifts non-full level of leaves to the end of the array
template<typename TYPE>
void permute_leaves(TYPE *dev_A, uint64_t n, uint64_t b, uint64_t numInternals, uint64_t numLeaves) {
    uint64_t r = numLeaves % b;     //number of leaves belonging to a non-full node
    uint64_t l = numLeaves - r;     //number of leaves belonging to full nodes
    uint64_t i = l/b;               //number of internals partitioning the leaf nodes

    #ifdef DEBUG
    printf("permute_leaves: r = %lu; l = %lu; i = %lu\n", r, l , i);
    #endif

    extended_equidistant_gather2<TYPE>(dev_A, l+i, b);

    shift_right_phaseOne<TYPE><<<BLOCKS, THREADS>>>(&dev_A[i], numLeaves + numInternals - i, numInternals - i);
    #ifdef DEBUG
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        printf("shift_right_phaseOne failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
    }
    #endif

    shift_right_phaseTwo<TYPE><<<BLOCKS, THREADS>>>(&dev_A[i], numLeaves + numInternals - i, numInternals - i);
    #ifdef DEBUG
    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        printf("shift_right_phaseTwo failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
    }
    #else
    cudaDeviceSynchronize();
    #endif
}

//permutes sorted array into level order b-tree layout for n = (b + 1)^d - 1
template<typename TYPE>
void permute(TYPE *dev_A, uint64_t n, uint64_t b, uint32_t d) {
    while (d > 1) {
        extended_equidistant_gather<TYPE>(dev_A, n, b, d);

        n /= b+1;
        d--;
    }
}

//Permutes dev_A into the implicit Btree level-order layout 
//Returns the time (in ms) to perform the permutation
//Assumes dev_A has already been initialized
template<typename TYPE>
double timePermuteBtree(TYPE *dev_A, uint64_t n, uint64_t b) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    uint32_t h = log10(n)/log10(b+1);
    if (n != pow(b+1, h+1) - 1) {       //non-full tree
        uint64_t numInternals = pow(b+1, h) - 1;
        uint64_t numLeaves = n - numInternals;

        #ifdef DEBUG
        printf("non-perfect B-tree\n");
        #endif

        permute_leaves<TYPE>(dev_A, n, b, numInternals, numLeaves);
        permute<TYPE>(dev_A, n - numLeaves, b, h);
    }
    else {    //full tree
        #ifdef DEBUG
        printf("perfect B-tree\n");
        #endif
        
        permute<TYPE>(dev_A, n, b, h+1);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double ms = ((end.tv_sec*1000000000. + end.tv_nsec) - (start.tv_sec*1000000000. + start.tv_nsec)) / 1000000.;   //millisecond
    return ms;
}
#endif