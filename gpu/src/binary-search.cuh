/*
 * Copyright 2018-2021 Kyle Berney
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

#ifndef BINARY_SEARCH_CUH
#define BINARY_SEARCH_CUH

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "common.cuh"

//Performs binary search on array A of size n
//Assumes the array is sorted
template<typename TYPE>
__device__ uint64_t binarySearch(TYPE *A, uint64_t n, TYPE query) {
    uint64_t mid;
    uint64_t left = 0;
    uint64_t right = n - 1;

    while (left <= right) {
        mid = (left + right + 1) / 2;       //left + ((right - left + 1) / 2);

        if (query == A[mid]) {
            return mid;
        }
        else if (query > A[mid]) {
            left = mid + 1;
        }
        else {
            right = mid - 1;
        }
    }
    return right;       //query not found
}

//Performs all of the queries given in the array queries
//Index of the queried items are saved in the answeres array
template<typename TYPE>
__global__ void searchAll(TYPE *A, uint64_t n, TYPE *queries, uint64_t numQueries) {
    int tid = threadIdx.x + blockIdx.x * THREADS;

    for (uint64_t i = tid; i < numQueries; i += BLOCKS*THREADS) {
        queries[i] = binarySearch<TYPE>(A, n, queries[i]);
    }
}

//Generates numQueries random queries and returns the milliseconds needed to perform the queries
template<typename TYPE>
float timeQuery(TYPE *A, TYPE *dev_A, uint64_t n, uint64_t numQueries) {
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    TYPE *queries = createRandomQueries<TYPE>(A, n, numQueries);              //array to store random queries to perform
    TYPE *dev_queries;
    cudaMalloc(&dev_queries, numQueries * sizeof(TYPE));

    #ifdef VERIFY
    TYPE *answers = (TYPE *)malloc(numQueries * sizeof(TYPE));    //array to store the answers (i.e., index of the queried item)
    #endif

    cudaMemcpy(dev_A, A, n * sizeof(TYPE), cudaMemcpyHostToDevice);                             //transfer A to GPU
    cudaMemcpy(dev_queries, queries, numQueries * sizeof(TYPE), cudaMemcpyHostToDevice);        //transfer queries to GPU

    cudaEventRecord(start);
    
    searchAll<TYPE><<<BLOCKS, THREADS>>>(dev_A, n, dev_queries, numQueries);
    #ifdef DEBUG
    cudaEventRecord(end);
    cudaError_t cudaerr = cudaEventSynchronize(end);
    if (cudaerr != cudaSuccess) {
        printf("searchAll failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
    }
    #else
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    #endif

    float ms;
    cudaEventElapsedTime(&ms, start, end);

    #ifdef VERIFY
    cudaMemcpy(answers, dev_queries, numQueries * sizeof(TYPE), cudaMemcpyDeviceToHost);        //transfer answers back to CPU
    bool correct = true;
    for (uint64_t i = 0; i < numQueries; i++) {
        if (answers[i] == n || A[answers[i]] != queries[i]) {
            #ifdef DEBUG
            printf("query = %lu; found = %lu\n", queries[i], A[answers[i]]);
            #endif
            correct = false;
        }
    }
    if (correct == false) printf("Searches failed!\n");
    else printf("Searches succeeded!\n");
    free(answers);
    #endif

    free(queries);
    cudaFree(dev_queries);

    return ms;
}
#endif