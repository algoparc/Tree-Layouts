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
    uint64_t i = n/2;

	for (uint64_t step = n/4 + 1; step >= 1; step /= 2) {
		if (query == A[i]) {
			return i;
		}
		else if (query > A[i]) {
			i += step;
		}
		else {
			i -= step;
		}
	}
	return i;
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

    #ifdef DEBUG
    uint64_t *answers = (uint64_t *)malloc(numQueries * sizeof(uint64_t));    //array to store the answers (i.e., index of the queried item)
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

    #ifdef DEBUG
    cudaMemcpy(answers, dev_queries, numQueries * sizeof(uint64_t), cudaMemcpyDeviceToHost);        //transfer answers back to CPU
    
    bool correct = true;
    for (uint64_t i = 0; i < numQueries; i++) {
        if (answers[i] == n || A[answers[i]] != queries[i]) {
            //printf("query = %lu; found = %lu\n", queries[i], A[answers[i]]);
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