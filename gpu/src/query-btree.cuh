/*
 * Copyright 2018-2020 Kyle Berney, Ben Karsin
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

#ifndef QUERY_BTREE_CUH
#define QUERY_BTREE_CUH

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "common.cuh"

//Searches given B-tree level order layout for the query'd element
//1 query per thread
//Returns index of query'd element (if found)
//Otherwise, returns n (element not found)
template<typename TYPE>
__device__ uint64_t searchBtree(TYPE *A, uint64_t n, uint64_t b, TYPE query) {
    TYPE current;
    uint64_t i = 0;
    uint64_t j;

    while (i < n) {
        for (j = 0; j < b; ++j) {   //linear search to find child pointer to follow
            current = A[i + j];

            if (query == current) {
                return i+j;
            }
            else if (query < current) {
                break;
            }
        }
        i = (i + 1)*(b + 1) + j*b - 1;
    }

    return n;   //error value, i.e., element not found
}

//Performs all of the queries given in the array queries
//index in A of the queried items are saved in the answers array
template<typename TYPE>
__global__ void searchAll(TYPE *A, uint64_t n, uint64_t b, TYPE *queries, uint64_t numQueries) {
    int tid = threadIdx.x + blockIdx.x * THREADS;

    for (uint64_t i = tid; i < numQueries; i += BLOCKS*THREADS) {
        queries[i] = searchBtree<TYPE>(A, n, b, queries[i]);
    }
}

//Generates numQueries random queries and returns the milliseconds needed to perform the queries on the given level-order btree layout
template<typename TYPE>
float timeQueryBtree(TYPE *A, TYPE *dev_A, uint64_t n, uint64_t b, uint64_t numQueries) {
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    TYPE *queries = createRandomQueries<TYPE>(A, n, numQueries);              //array to store random queries to perform
    TYPE *dev_queries;
    cudaMalloc(&dev_queries, numQueries * sizeof(TYPE));

    #ifdef VERIFY
    uint64_t *answers = (uint64_t *)malloc(numQueries * sizeof(uint64_t));    //array to store the answers (i.e., index of the queried item)
    #endif

    cudaMemcpy(dev_A, A, n * sizeof(TYPE), cudaMemcpyHostToDevice);                             //transfer A to GPU
    cudaMemcpy(dev_queries, queries, numQueries * sizeof(TYPE), cudaMemcpyHostToDevice);        //transfer queries to GPU

    cudaEventRecord(start);
    
    searchAll<TYPE><<<BLOCKS, THREADS>>>(dev_A, n, b, dev_queries, numQueries);
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
    cudaMemcpy(answers, dev_queries, numQueries * sizeof(uint64_t), cudaMemcpyDeviceToHost);        //transfer answers back to CPU
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