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

#ifndef VEB_INVOLUTIONS_CUH
#define VEB_INVOLUTIONS_CUH

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#include "common.cuh"
#include "involutions.cuh"

//Permutes the given sorted array into the van Emde Boas tree layout
//Recurses on the GPU
//Assumes a single block is permuting the array
template<typename TYPE>
__device__ void permutevEB_gpu(TYPE *A, uint64_t n, uint32_t d) {
    if (d > 2) {
        uint32_t leaf_d = (d - 2)/2 + 1;        //floor((d - 2)/2) + 1
        uint32_t root_d = d - leaf_d;           //ceil((d - 2)/2.) + 1

        uint64_t r = pow(2, root_d) - 1;        //number of elements in root subtree
        uint64_t l = pow(2, leaf_d) - 1;        //number of elements in leaf subtrees

        uint32_t h = log2((double)n)/log2((double)l+1);
        if (n != pow(l+1, h+1) - 1) {
            //printf("non-perfect B-tree of height %ld\n", h);
            unshuffle_dk_blocks<TYPE>(A-1, n+1, l+1, n+1);
        }
        else {
            //printf("perfect B-tree of height %ld\n", h);
            unshuffle_blocks<TYPE>(A, n, l+1, n, h+1);
        }
        shuffle_dk_blocks<TYPE>(&A[r], n-r, l, n-r);

        permutevEB_gpu<TYPE>(A, r, root_d);       //recurse on root subtree
        __syncthreads();
        for (uint32_t i = 0; i < r+1; ++i) {    //recurse on leaf subtrees
            permutevEB_gpu<TYPE>(&A[r + i*l], l, leaf_d);
            __syncthreads();
        }
    }
    else if (d == 2) {      //d == 2, single swap
        if (threadIdx.x == 0) {
            TYPE temp;
            temp = A[0];
            A[0] = A[1];
            A[1] = temp;
        }
    }
}

//Permutes the the given array consisting of a root subtree and leaf subtrees into the van Emde Boas tree layout
//1 block per subtree
template<typename TYPE>
__global__ void permutevEB_blocks(TYPE *A, uint32_t root_d, uint32_t leaf_d, uint64_t r, uint64_t l) {
    if (blockIdx.x == 0) {      //0-th block permutes root subtree
        permutevEB_gpu<TYPE>(A, r, root_d);
    }
    else {
        TYPE *temp_A = &A[r];
        //uint64_t num_trees = (n - r)/l;
        for (uint32_t i = blockIdx.x-1; i < r+1; i += gridDim.x-1) {      //r+1 leaf subtrees to permute, 1 block per
            permutevEB_gpu<TYPE>(&temp_A[i*l], l, leaf_d);
        }
    }
}

//Performs a single round of the van Emde Boas tree layout permutation via B-tree involutions
template<typename TYPE>
void permutevEB_grid(TYPE *dev_A, uint64_t n, uint32_t root_d, uint32_t leaf_d, uint64_t r, uint64_t l) {
    uint32_t h = log2(n)/log2(l+1);
    if (n != pow(l+1, h+1) - 1) {
        //printf("non-perfect B-tree of height %ld\n", h);
        shuffle_dk_phaseTwo<TYPE><<<BLOCKS, THREADS>>>(dev_A - 1, l+1, n+1);
        #ifdef DEBUG
	    cudaError_t cudaerr = cudaDeviceSynchronize();
	    if (cudaerr != cudaSuccess) {
	        printf("shuffle_dk_phaseTwo failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
	    }
	    #endif

        shuffle_dk_phaseOne<TYPE><<<BLOCKS, THREADS>>>(dev_A - 1, l+1, n+1);
        #ifdef DEBUG
	    cudaerr = cudaDeviceSynchronize();
	    if (cudaerr != cudaSuccess) {
	        printf("shuffle_dk_phaseOne failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
	    }
	    #endif
    }
    else {
        //printf("perfect B-tree of height %ld\n", h);
        unshuffle_phaseOne<TYPE><<<BLOCKS, THREADS>>>(dev_A, l+1, n, h+1);
        #ifdef DEBUG
	    cudaError_t cudaerr = cudaDeviceSynchronize();
	    if (cudaerr != cudaSuccess) {
	        printf("unshuffle_phaseOne failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
	    }
	    #endif

        unshuffle_phaseTwo<TYPE><<<BLOCKS, THREADS>>>(dev_A, l+1, n, h+1);
        #ifdef DEBUG
	    cudaerr = cudaDeviceSynchronize();
	    if (cudaerr != cudaSuccess) {
	        printf("unshuffle_phaseTwo failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
	    }
	    #endif
    }
    shuffle_dk_phaseOne<TYPE><<<BLOCKS, THREADS>>>(&dev_A[r], l, n-r);
    #ifdef DEBUG
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        printf("shuffle_dk_phaseOne failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
    }
    #endif
    shuffle_dk_phaseTwo<TYPE><<<BLOCKS, THREADS>>>(&dev_A[r], l, n-r);
    #ifdef DEBUG
    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        printf("shuffle_dk_phaseTwo failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
    }
    #else
    cudaDeviceSynchronize();
    #endif
}

//Permutes sorted array into the van Emde Boas tree layout
//B = # of leaf elements per leaf subtree, i.e., paramter l (in the below code)
//The top subtree has a height of ceil{(h - 1)/2} and leaf subtrees have height of floor{(h - 1)/2}
//Recurses on the GPU
template<typename TYPE>
void permutevEB(TYPE *dev_A, uint64_t n, uint32_t d) {
    if (n == 1) return;

    uint32_t leaf_d = (d - 2)/2 + 1;        //floor((d - 2)/2) + 1
    uint32_t root_d = d - leaf_d;           //ceil((d - 2)/2.) + 1

    uint64_t r = pow(2, root_d) - 1;        //number of elements in root subtree
    uint64_t l = pow(2, leaf_d) - 1;        //number of elements in leaf subtrees

    permutevEB_grid<TYPE>(dev_A, n, root_d, leaf_d, r, l);
    permutevEB_blocks<TYPE><<<BLOCKS, THREADS>>>(dev_A, root_d, leaf_d, r, l);
    #ifdef DEBUG
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        printf("permutevEB_blocks failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
    }
    #else
    cudaDeviceSynchronize();
    #endif
}

//Permutes sorted array into the van Emde Boas tree layout via Level-order B-tree involutions
//B = # of leaf elements per leaf subtree, i.e., paramter l (in the below code)
//The top subtree has a height of ceil{(h - 1)/2} and leaf subtrees have height of floor{(h - 1)/2}
template<typename TYPE>
void permutevEB_cpuRecursive(TYPE *dev_A, uint64_t n, uint32_t d) {
    if (n == 1) return;

    uint32_t leaf_d = (d - 2)/2 + 1;        //floor((d - 2)/2) + 1
    uint32_t root_d = d - leaf_d;           //ceil((d - 2)/2.) + 1

    uint64_t r = pow(2, root_d) - 1;        //number of elements in root subtree
    uint64_t l = pow(2, leaf_d) - 1;        //number of elements in leaf subtrees

    int blocks, threads;
    if (d > 8) {                    //grid level: n > 2^8 - 1 = 255
        blocks = BLOCKS;
        threads = THREADS;
    }
    else if (d > 5) {               //block level: n > 2^5 - 1 = 31
        blocks = 1;
        threads = (n + 1)/WARPS;
    }
    else {                          //warp level: n > 1
        blocks = 1;
        threads = 32;
    }

    uint32_t h = log2(n)/log2(l+1);
    if (n != pow(l+1, h+1) - 1) {
        //printf("non-perfect B-tree of height %ld\n", h);
        shuffle_dk_phaseTwo<TYPE><<<blocks, threads>>>(dev_A - 1, l+1, n+1);
        #ifdef DEBUG
        cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess) {
            printf("shuffle_dk_phaseTwo failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
        }
        #endif

        shuffle_dk_phaseOne<TYPE><<<blocks, threads>>>(dev_A - 1, l+1, n+1);
        #ifdef DEBUG
        cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess) {
            printf("shuffle_dk_phaseOne failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
        }
        #endif
    }
    else {
        //printf("perfect B-tree of height %ld\n", h);
        unshuffle_phaseOne<TYPE><<<blocks, threads>>>(dev_A, l+1, n, h+1);
        #ifdef DEBUG
        cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess) {
            printf("unshuffle_phaseOne failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
        }
        #endif

        unshuffle_phaseTwo<TYPE><<<blocks, threads>>>(dev_A, l+1, n, h+1);
        #ifdef DEBUG
        cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess) {
            printf("unshuffle_phaseTwo failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
        }
        #endif
    }
    shuffle_dk_phaseOne<TYPE><<<blocks, threads>>>(&dev_A[r], l, n-r);
    #ifdef DEBUG
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        printf("shuffle_dk_phaseOne failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
    }
    #endif
    shuffle_dk_phaseTwo<TYPE><<<blocks, threads>>>(&dev_A[r], l, n-r);
    #ifdef DEBUG
    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        printf("shuffle_dk_phaseTwo failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
    }
    #else
    cudaDeviceSynchronize();
    #endif

    permutevEB<TYPE>(dev_A, r, root_d);               //recurse on root subtree
    uint32_t numLeafTrees = (n - r)/l;
    for (int i = 0; i < numLeafTrees; i++) {
        permutevEB<TYPE>(&dev_A[r + i*l], l, leaf_d);        //recurse on i-th leaf subtree
    }
}

//Permutes dev_A into the implicit van Emde Boas tree layout 
//Returns the time (in ms) to perform the permutation
//Assumes dev_A has already been initialized
template<typename TYPE>
double timePermutevEB(TYPE *dev_A, uint64_t n) {
	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC, &start);

    uint32_t h = log2(n);
    if (n != pow(2, h+1) - 1) {     //non-full tree
        printf("non-perfect vEB tree ==> NOT YET IMPLEMENTED!\n");
        return 0.;
    }
    else {    //full tree
        permutevEB<TYPE>(dev_A, n, h+1);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double ms = ((end.tv_sec*1000000000. + end.tv_nsec) - (start.tv_sec*1000000000. + start.tv_nsec)) / 1000000.;		//millisecond
    return ms;
}
#endif