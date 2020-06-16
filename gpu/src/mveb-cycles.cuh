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

#ifndef MVEB_CYCLES_CUH
#define MVEB_CYCLES_CUH

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#include "common.cuh"
#include "cycles.cuh"

//1 block per partition of m = 2^8 - 1 = 255
template<typename TYPE>
__global__ void permutevEB_balanced_basecase8(TYPE *A, uint64_t n) {
    TYPE *block_a;
    TYPE *warp_a;
    
    int wid = threadIdx.x/WARPS;            //warp id within a block
    int num_warps = blockDim.x/WARPS;       //number of warps per block

    int tid = threadIdx.x % WARPS;          //thread id, within a warp

    //m = 255, d = 8, k = 15
    for (uint64_t i = blockIdx.x; i < n/255; i += gridDim.x) {
        block_a = &A[i*255];                            //start of the partition for the block

        equidistant_gather_blocks<TYPE>(block_a, 15, 15);

        //m = 15, d = 4, k = 3
        //17 partitions per block
        for (int j = wid; j < 17; j += num_warps) {
            warp_a = &block_a[j*15];                    //start of the partition for the warp

            equidistant_gather_warps<TYPE>(warp_a, 3, 3);

            //m = 3, d = 2, k = 1
            //5 partitions per warp
            if (tid < 5) {
                TYPE *thread_a = &warp_a[3*tid];        //start of the partition for the thread
                
                TYPE temp = thread_a[0];
                thread_a[0] = thread_a[1];
                thread_a[1] = temp;
            }
        }
    }
}

//1 warp per partition of m = 2^4 - 1 = 15
template<typename TYPE>
__global__ void permutevEB_balanced_basecase4(TYPE *A, uint64_t n) {
    TYPE *a;
    
    int wid = (threadIdx.x + blockIdx.x*blockDim.x)/WARPS;      //global warp id
    int num_warps = (gridDim.x*blockDim.x)/WARPS;               //total number of warps

    int tid = threadIdx.x % WARPS;                              //thread id, within a warp

    //m = 15, d = 4, k = 3
    for (uint64_t i = wid; i < n/15; i += num_warps) {
        a = &A[i*15];                               //start of the partition for the warp

        equidistant_gather_warps<TYPE>(a, 3, 3);

        //m = 3, d = 2, k = 1
        //5 partitions per warp
        if (tid < 5) {
            TYPE *thread_a = &a[3*tid];        //start of the partition for the thread
                
            TYPE temp = thread_a[0];
            thread_a[0] = thread_a[1];
            thread_a[1] = temp;
        }
    }
}

//1 thread per partition of m = 2^2 - 1 = 3
template<typename TYPE>
__global__ void permutevEB_balanced_basecase2(TYPE *A, uint64_t n) {
    TYPE *a;
    TYPE temp;

    int tid = threadIdx.x + blockIdx.x*blockDim.x;          //global thread id
    int num_threads = gridDim.x*blockDim.x;                 //total number of threads

    //m = 3, d = 2, k = 1
    for (uint64_t i = tid; i < n/3; i += num_threads) {
        a = &A[i*3];        //start of the partition for the thread

        temp = a[0];
        a[0] = a[1];
        a[1] = temp;
    }
}

//Permutes each partition of m elements into the van Emde Boas tree layout
//Assumes n is a multiple of m = 2^d - 1
//Assumes d is a power of 2
template<typename TYPE>
void permutevEB_balanced(TYPE *dev_A, uint64_t n, uint64_t m, uint32_t d) {
    if (d > 8) {
        while (d > 8) {
            int blocks = (BLOCKS > n/m) ? BLOCKS : n/m;
            uint64_t k = (uint64_t)sqrt(m);       

            equidistant_gather_partitions_phaseOne<TYPE><<<blocks, THREADS>>>(dev_A, n, m, k, k);
            #ifdef DEBUG
            cudaError_t cudaerr = cudaDeviceSynchronize();
            if (cudaerr != cudaSuccess) {
                printf("equidistant_gather_partitions_phaseOne failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
            }
            #endif

            equidistant_gather_partitions_phaseTwo<TYPE><<<blocks, THREADS>>>(dev_A, n, m, k, k);
            #ifdef DEBUG
            cudaerr = cudaDeviceSynchronize();
            if (cudaerr != cudaSuccess) {
                printf("equidistant_gather_partitions_phaseTwo failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
            }
            #else
            cudaDeviceSynchronize();
            #endif

            m = k;
            d /= 2;
        }

        permutevEB_balanced_basecase8<TYPE><<<n/m, THREADS>>>(dev_A, n);      //1 block per partition of m = 2^8 - 1 = 255 elements
        #ifdef DEBUG
        cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess) {
            printf("permutevEB_balanced_basecase8 failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
        }
        #else
        cudaDeviceSynchronize();
        #endif
    }
    else if (d == 8) {
        permutevEB_balanced_basecase8<TYPE><<<n/m, THREADS>>>(dev_A, n);      //1 block per partition of m = 2^8 - 1 = 255 elements
        #ifdef DEBUG
        cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess) {
            printf("permutevEB_balanced_basecase8 failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
        }
        #else
        cudaDeviceSynchronize();
        #endif
    }
    else if (d == 4) {
        //1 warp per partition of m = 2^4 - 1 = 15 elements
        if (n/m <= THREADS/WARPS) {     //1 block
            int threads = (n/m)*WARPS;
            permutevEB_balanced_basecase4<TYPE><<<1, threads>>>(dev_A, n);
            #ifdef DEBUG
            cudaError_t cudaerr = cudaDeviceSynchronize();
            if (cudaerr != cudaSuccess) {
                printf("permutevEB_balanced_basecase4 failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
            }
            #else
            cudaDeviceSynchronize();
            #endif
        }
        else {
            int blocks = (n/m)/(THREADS/WARPS);
            permutevEB_balanced_basecase4<TYPE><<<blocks, THREADS>>>(dev_A, n);
            #ifdef DEBUG
            cudaError_t cudaerr = cudaDeviceSynchronize();
            if (cudaerr != cudaSuccess) {
                printf("permutevEB_balanced_basecase4 failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
            }
            #else
            cudaDeviceSynchronize();
            #endif
        }
    }
    else if (d == 2) {
        //1 thread per partition of m = 2^2 - 1 = 3 elements
        if (n/m <= THREADS) {       //1 block
            permutevEB_balanced_basecase2<TYPE><<<1, n/m>>>(dev_A, n);        //TODO: round up threads to the nearest 32?
            #ifdef DEBUG
            cudaError_t cudaerr = cudaDeviceSynchronize();
            if (cudaerr != cudaSuccess) {
                printf("permutevEB_balanced_basecase2 failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
            }
            #else
            cudaDeviceSynchronize();
            #endif
        }
        else {
            int blocks = (n/m)/THREADS;
            permutevEB_balanced_basecase2<TYPE><<<blocks, THREADS>>>(dev_A, n);
            #ifdef DEBUG
            cudaError_t cudaerr = cudaDeviceSynchronize();
            if (cudaerr != cudaSuccess) {
                printf("permutevEB_balanced_basecase2 failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
            }
            #else
            cudaDeviceSynchronize();
            #endif
        }
    }
}

//Permutes the array of size n into the van Emde Boas tree layout
//Assumes n = 2^d - 1
template<typename TYPE>
void permutevEB(TYPE *dev_A, uint64_t n, uint32_t d) {
    if (n == 1) return;

    uint32_t leaf_d = (d - 2)/2 + 1;
    float log_leaf = log2((float)leaf_d);
    if (log_leaf - ((int)log_leaf) != 0) {      //Not a perfectly balanced vEB, i.e., d is not a power of 2
        leaf_d = pow(2, ceil(log_leaf));
    }

    uint32_t root_d = d - leaf_d;

    uint64_t r = pow(2, root_d) - 1;
    uint64_t l = pow(2, leaf_d) - 1;

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

    if (r <= l) {
        equidistant_gather_phaseOne<TYPE><<<blocks, threads>>>(dev_A, r, l);
        #ifdef DEBUG
        cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess) {
            printf("equidistant_gather_phaseOne failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
        }
        #endif

        equidistant_gather_phaseTwo<TYPE><<<blocks, threads>>>(dev_A, r, l);
        #ifdef DEBUG
        cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess) {
            printf("equidistant_gather_phaseTwo failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
        }
        #else
        cudaDeviceSynchronize();
        #endif
    }
    else {      //r = 2*l + 1
        equidistant_gather_phaseOne<TYPE><<<blocks, threads>>>(dev_A, l, l);
        #ifdef DEBUG
        cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess) {
            printf("equidistant_gather_phaseOne failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
        }
        #endif

        equidistant_gather_phaseTwo<TYPE><<<blocks, threads>>>(dev_A, l, l);
        #ifdef DEBUG
        cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess) {
            printf("equidistant_gather_phaseTwo failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
        }
        #endif

        equidistant_gather_phaseOne<TYPE><<<blocks, threads>>>(&dev_A[n/2 + 1], l, l);
        #ifdef DEBUG
        cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess) {
            printf("equidistant_gather_phaseOne failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
        }
        #endif

        equidistant_gather_phaseTwo<TYPE><<<blocks, threads>>>(&dev_A[n/2 + 1], l, l);
        #ifdef DEBUG
        cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess) {
            printf("equidistant_gather_phaseTwo failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
        }
        #endif

        shift_right_phaseOne<TYPE><<<blocks, threads>>>(&dev_A[l], (l+1)*(l+1), l+1);
        #ifdef DEBUG
        cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess) {
            printf("shift_right_phaseOne failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
        }
        #endif

        shift_right_phaseTwo<TYPE><<<blocks, threads>>>(&dev_A[l], (l+1)*(l+1), l+1);
        #ifdef DEBUG
        cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess) {
            printf("shift_right_phaseTwo failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
        }
        #else
        cudaDeviceSynchronize();
        #endif
    }

    permutevEB_balanced<TYPE>(&dev_A[r], n-r, l, leaf_d);      //permute leaf subtrees

    //TODO: check if root_d is a power of 2
    permutevEB<TYPE>(dev_A, r, root_d);        //recurse on root subtree
}

//Permutes dev_A into the implicit modified van Emde Boas tree layout 
//Returns the time (in ms) to perform the permutation
//Assumes dev_A has already been initialized
template<typename TYPE>
double timePermutevEB(TYPE *dev_A, uint64_t n) {
	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC, &start);

    uint32_t h = log2((double)n);
    if (n != pow(2, h+1) - 1) {     //non-full tree
        printf("non-perfect vEB tree ==> NOT YET IMPLEMENTED!\n");
        return 0.;
    }
    else {    //full tree
        float log_d = log2((float)(h+1));

        if (log_d - ((int)log_d) == 0) permutevEB_balanced<TYPE>(dev_A, n, n, h+1);     //d is a power of 2 <==> n is a power of power of 2 minus 1
        else {
            permutevEB<TYPE>(dev_A, n, h+1);
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double ms = ((end.tv_sec*1000000000. + end.tv_nsec) - (start.tv_sec*1000000000. + start.tv_nsec)) / 1000000.;		//millisecond
    return ms;
}
#endif
