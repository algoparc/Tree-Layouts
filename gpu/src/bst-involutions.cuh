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

#ifndef BST_INVOLUTIONS_CUH
#define BST_INVOLUTIONS_CUH

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>

#include "common.cuh"
#include "involutions.cuh"

//First involution used in BST permutation
template<typename TYPE>
__global__ void phaseOne(TYPE *A, uint64_t n, uint64_t d) {
    int tid = threadIdx.x + blockIdx.x * THREADS;
    uint64_t j;
    TYPE temp;

    for (uint64_t i = tid; i < n; i += THREADS*BLOCKS) {
        j = rev_d(i+1, d) - 1;

        if (i < j) {
            temp = A[i];
            A[i] = A[j];
            A[j] = temp;
        }
    }
}

//Second involution used in BST permutation
template<typename TYPE>
__global__ void phaseTwo(TYPE *A, uint64_t n, uint64_t d) {
    int tid = threadIdx.x + blockIdx.x * THREADS;
    uint64_t j;
    TYPE temp;

    for (uint64_t i = tid; i < n; i += THREADS*BLOCKS) {
        if (i+1 > n/2) {        //leafs
            j = rev_b(i+1, d, d-1) - 1;
        }
        else {      //internals
            j = rev_b(i+1, d, d - (__clzll(i+1) - (64 - d) + 1)) - 1;
        }

        if (i < j) {
            temp = A[i];
            A[i] = A[j];
            A[j] = temp;
        }
    }
}

//Permutes dev_A into the implicit BST level-order layout 
//Returns the time (in ms) to perform the permutation
//Assumes dev_A has already been initialized
template<typename TYPE>
float timePermuteBST(TYPE *dev_A, uint64_t n) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    uint32_t d = log2(n) + 1;

    phaseOne<TYPE><<<BLOCKS, THREADS>>>(dev_A, n, d);
    #ifdef DEBUG
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        printf("phaseOne failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
    }
    #endif

    phaseTwo<<<BLOCKS, THREADS>>>(dev_A, n, d);
    #ifdef DEBUG
    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        printf("phaseTwo failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
    }
    #else
    cudaDeviceSynchronize();
    #endif

    clock_gettime(CLOCK_MONOTONIC, &end);
    double ms = ((end.tv_sec*1000000000. + end.tv_nsec) - (start.tv_sec*1000000000. + start.tv_nsec)) / 1000000.;   //millisecond
    return ms;
}
#endif