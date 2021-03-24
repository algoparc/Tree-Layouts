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

#ifndef COMMON_CUH
#define COMMON_CUH

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include <time.h>

#include "params.cuh"

void printA(uint64_t *A, uint64_t n);
void printA(uint32_t *A, uint64_t n);
void printTimings(uint64_t n, double time[ITERS]);
void printQueryTimings(uint64_t n, uint64_t q, double time[ITERS]);

//Populates the given array of size with 1, 2, ..., n casted to the given generic type
template<typename TYPE>
void initSortedList(TYPE *data, uint64_t n) {
    #pragma omp parallel for
    for (uint64_t i = 0; i < n; i++) {
        data[i] = (TYPE)(i+1);
    }
}

//Creates random queries for the given data array of n elements
//Returns a pointer to an array containing the generated random queries of size numQueries
template<typename TYPE>
TYPE *createRandomQueries(TYPE *data, uint64_t n, uint64_t numQueries) {
    TYPE *queries = (TYPE *)malloc(numQueries * sizeof(TYPE));

    #ifdef DEBUG
    srand(0);
    #else
    srand(time(NULL));
    #endif
  
    #pragma omp parallel for
    for (uint64_t i = 0; i < numQueries; i++) {
        queries[i] = (TYPE)data[rand() % n];
    }

    return queries;
}

//Performs the first phase of shifting n contiguous elements by k to the right via array reversals
//Also the second phase for shift left by k
//1 grid on the entire array
template<typename TYPE>
__global__ void shift_right_phaseOne(TYPE *A, uint64_t n, uint64_t k) {
    uint64_t j;
    TYPE temp;

    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int num_threads = gridDim.x * blockDim.x;

    //stage 1: reverse whole array
    for (uint64_t i = tid; i < n/2; i += num_threads) {
        j = n - i - 1;

        temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }
}

//Performs the second phase of shifting n contiguous elements by k to the right via array reversals
//Also the first phase for shift left by k 
//1 grid on the entire array
template<typename TYPE>
__global__ void shift_right_phaseTwo(TYPE *A, uint64_t n, uint64_t k) {
    uint64_t j;
    TYPE temp;

    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int num_threads = gridDim.x * blockDim.x;

    //stage 2: reverse first k elements & last (n - k) elements
    for (uint64_t i = tid; i < k/2; i += num_threads) {
        j = k - i - 1;

        temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }

    for (uint64_t i = tid + k; i < (n + k)/2; i += num_threads) {
        j = n - (i - k) - 1;

        temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }
}
#endif