#ifndef QUERY_VEB_CUH
#define QUERY_VEB_CUH

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include "common.cuh"

struct vEB_table {
	uint64_t L;			//size of the bottom/leaf tree
	uint64_t R;			//size of the corresponding top/root tree
	uint32_t D;			//depth of the root of the corresponding top/root tree
};

void buildTable(vEB_table *table, uint64_t n, uint32_t d, uint32_t root_depth);

//Searches given Btree layout for the query'd element using the table composed of L, R, and D
//Assumes L, R, and D have been initialized via buildTable()
//pos is a pointer to a shared memory array of size d * THREADS
    //which stores the 1-indexed position of the node visited during the
    //query at current depth in the vEB tree for all threads in the thread block
//Returns index of query'd element (if found)
//Otherwise, returns n (element not found)
template<typename TYPE>
__forceinline__ __device__ uint64_t searchvEB(TYPE *A, vEB_table *table, uint64_t n, uint32_t d, TYPE query, uint64_t *pos) {
    int tid = threadIdx.x;      //thread id within the thread block
    TYPE current;

    uint32_t current_d = 0;
    uint64_t i = 1;         //1-indexed position of the current node in a BFS (i.e, level order) binary search tree
    pos[tid] = 1;

    uint64_t index = 0;     //0-indexed position of the current node in the vEB tree
    while (index < n) {
        current = A[index];

        if (query == current) {
            return index;
        }

        i = 2*i + (query > current);
        current_d++;

        pos[tid + blockDim.x*current_d] = pos[tid + blockDim.x*table[current_d].D] + table[current_d].R + (i & table[current_d].R) * table[current_d].L;

        index = pos[tid + blockDim.x*current_d] - 1;
    }

    return n;
}

//Performs all of the queries given in the array queries
//index in A of the queried items are saved in the answers array
template<typename TYPE>
__global__ void searchAll(TYPE *A, vEB_table *table, uint64_t n, uint32_t d, TYPE *queries, uint64_t numQueries) {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int num_threads = gridDim.x*blockDim.x;
    extern __shared__ uint64_t pos[];

    for (uint64_t i = tid; i < numQueries; i += num_threads) {
        queries[i] = searchvEB(A, table, n, d, queries[i], pos);
    }
}

//Generates numQueries random queries and returns the milliseconds needed to perform the queries on the given vEB tree layout
//Assumes dev_table have already been initialized and transferred onto the GPU
template<typename TYPE>
float timeQueryvEB(TYPE *A, TYPE *dev_A, vEB_table *dev_table, uint64_t n, uint32_t d, uint64_t numQueries) {
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

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if ((d * sizeof(uint64_t) * THREADS) <= prop.sharedMemPerBlock) {
        cudaEventRecord(start);
        searchAll<<<BLOCKS, THREADS, (size_t)(d * sizeof(uint64_t) * THREADS)>>>(dev_A, dev_table, n, d, dev_queries, numQueries);
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
    }
    else {
        int blocks = BLOCKS/2;
        int threads = THREADS/2;
        while ((d * sizeof(uint64_t) * threads) / 1024. > prop.sharedMemPerBlock && threads > WARPS) {
            blocks /= 2;
            threads /= 2;
        }

        cudaEventRecord(start);
        searchAll<<<blocks, threads, (size_t)(d * sizeof(uint64_t) * threads)>>>(dev_A, dev_table, n, d, dev_queries, numQueries);
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
    }

    float ms;
    cudaEventElapsedTime(&ms, start, end);

    #ifdef DEBUG
    cudaMemcpy(answers, dev_queries, numQueries * sizeof(uint64_t), cudaMemcpyDeviceToHost);        //transfer answers back to CPU
    bool correct = true;
    for (uint64_t i = 0; i < numQueries; i++) {
        if (answers[i] == n || A[answers[i]] != queries[i] || answers[i] == n) {
            //printf("query = %lu; A[%lu] = %lu\n", queries[i], answers[i], A[answers[i]]);
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