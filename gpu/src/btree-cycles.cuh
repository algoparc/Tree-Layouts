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

//Performs the first phase of the equidistant gather chunks on each partition
//Number of elements in each partition is ((b+1)^d + 1)*c, except the first partition which has 1 less element
//Permutes ((b+1)^d - 1)*c elements in each partition, i.e., ignores the first c elements in each partition
//Assumes m = ((b+1)^d)*c
//Assumes gridDim.x >= num_partitions
//Multple blocks per partition ==> 1 block per cycle
template<typename TYPE>
__global__ void equidistant_gather_chunks_partitions_phaseOne(TYPE *A, uint64_t m, uint64_t b, uint64_t c, uint32_t num_partitions) {
    int blocks = gridDim.x / num_partitions;        //number of blocks per unshuffle on m elements
    int pid = blockIdx.x / blocks;                  //partition id, i.e., which partition of m elements the current block permutes
    int bid = blockIdx.x % blocks;                  //block id within the partition

    uint64_t idx = pid*m + c;
    if (idx != 0) --idx;
    TYPE *a = &A[idx];

    for (uint64_t i = bid; i < b; i += blocks) {            //1 block per cycle
        for (uint64_t t = threadIdx.x; t < c; t += blockDim.x) {
            TYPE temp1, temp2;

            temp1 = a[i*c + t];
            a[i*c + t] = a[((i+1)*(b+1) - 1)*c + t];
            temp2 = temp1;

            for (uint64_t j = (i+b)*c; j < ((i+1)*(b+1) - 1)*c; j += b*c) {
                temp1 = a[j + t];
                a[j + t] = temp2;
                temp2 = temp1;
            }
            a[((i+1)*(b+1) - 1)*c + t] = temp2;
        }
    }
}

//Performs the first phase of the equidistant gather chunks on each partition
//Number of elements in each partition is ((b+1)^d + 1)*c, except the first partition which has 1 less element
//Permutes c((b+1)^d - 1) elements in each partition, i.e., ignores the first c elements in each partition
//Assumes m = ((b+1)^d)*c
//Assumes gridDim.x >= num_partitions
//Multiple blocks per partition ==> 1 block per shift
template<typename TYPE>
__global__ void equidistant_gather_chunks_partitions_phaseTwo(TYPE *A, uint64_t m, uint64_t b, uint64_t c, uint32_t num_partitions) {
    int blocks = gridDim.x / num_partitions;                            //number of blocks per unshuffle on m elements
    int pid = blockIdx.x / blocks;                                      //partition id, i.e., which partition of m elements the current block permutes
    int bid = blockIdx.x % blocks;                                      //block id within the partition

    uint64_t idx = pid*m + c;
    if (idx != 0) --idx;
    TYPE *a = &A[idx];

    for (uint64_t i = bid; i < b; i += blocks) {      //1 block per shift
        uint64_t kr = (b - i)*c;
        uint64_t kl = i*c;

        if (kr <= kl && kr != 0 && kr != b*c) {
            shift_right_blocks<TYPE>(&a[(i+1)*b*c], b*c, kr);
        }
        else if (kl != 0 && kl != b*c) {
            shift_left_blocks<TYPE>(&a[(i+1)*b*c], b*c, kl);
        }
    }
}

//Performs the equidistant gather chunks on each partition
//Number of elements in each partition is ((b+1)^d + 1)*c, except the first partition which has 1 less element
//Permutes ((b+1)^d - 1)*c elements in each partition, i.e., ignores the first c elements in each partition
//Assumes m = ((b+1)^d)*c
//Asumes total number of blocks launched is equal to num_partitions
//1 block per partition ==> 1 warp per cycle/shift
template<typename TYPE>
__global__ void equidistant_gather_chunks_partitions_blocks(TYPE *A, uint64_t m, uint64_t b, uint64_t c, uint32_t num_partitions) {
    int wid = threadIdx.x / WARPS;          //warp id within block
    int tid = threadIdx.x % WARPS;          //thread id within warp
    int num_warps = blockDim.x / WARPS;     //number of warps per block

    uint64_t idx = blockIdx.x*m + c;
    if (idx != 0) --idx;
    TYPE *a = &A[idx];

    for (uint64_t i = wid; i < b; i += num_warps) {            //1 warp per cycle
        for (uint64_t t = tid; t < c; t += WARPS) {
            TYPE temp1, temp2;

            temp1 = a[i*c + t];
            a[i*c + t] = a[((i+1)*(b+1) - 1)*c + t];
            temp2 = temp1;

            for (uint64_t j = (i+b)*c; j < ((i+1)*(b+1) - 1)*c; j += b*c) {
                temp1 = a[j + t];
                a[j + t] = temp2;
                temp2 = temp1;
            }
            a[((i+1)*(b+1) - 1)*c + t] = temp2;
        }
    }
    __syncthreads();

    for (uint64_t i = wid; i < b; i += num_warps) {      //1 warp per shift
        uint64_t kr = (b - i)*c;
        uint64_t kl = i*c;

        if (kr <= kl && kr != 0 && kr != b*c) {
            shift_right_warps<TYPE>(&a[(i+1)*b*c], b*c, kr);
        }
        else if (kl != 0 && kl != b*c) {
            shift_left_warps<TYPE>(&a[(i+1)*b*c], b*c, kl);
        }
    }
}

//Performs the equidistant gather chunks on each partition
//Number of elements in each partition is ((b+1)^d + 1)*c, except the first partition which has 1 less element
//Permutes ((b+1)^d - 1)*c elements in each partition, i.e., ignores the first c elements in each partition
//Assumes m = ((b+1)^d)*c
//Assumes total number of warps launched is equal to num_partitions
//1 warp per partition ==> 1 thread per cycle/shift
template<typename TYPE>
__global__ void equidistant_gather_chunks_partitions_warps(TYPE *A, uint64_t n, uint64_t m, uint64_t b, uint64_t c, uint32_t num_partitions) {
    int wid = (threadIdx.x + blockIdx.x*blockDim.x)/WARPS;      //global warp id
    int tid = threadIdx.x % WARPS;                              //thread id within warp

    uint64_t idx = wid*m + c;
    if (idx != 0) --idx;
    TYPE *a = &A[idx];

    //TODO: For small b values, this loop needs to be optimized so that there are WARPS/c threads per cycle
    //I.e., change loop to perform coalsesced global accesses
    for (uint64_t i = tid; i < b; i += WARPS) {            //1 thread per cycle
        for (uint64_t t = 0; t < c; ++t) {
            TYPE temp1, temp2;

            temp1 = a[i*c + t];
            a[i*c + t] = a[((i+1)*(b+1) - 1)*c + t];
            temp2 = temp1;

            for (uint64_t j = (i+b)*c; j < ((i+1)*(b+1) - 1)*c; j += b*c) {
                temp1 = a[j + t];
                a[j + t] = temp2;
                temp2 = temp1;
            }
            a[((i+1)*(b+1) - 1)*c + t] = temp2;
        }
    }

    for (uint64_t i = tid; i < b; i += WARPS) {      //1 thread per shift
        uint64_t kr = (b - i)*c;
        uint64_t kl = i*c;

        if (kr <= kl && kr != 0 && kr != b*c) {
            shift_right_threads<TYPE>(&a[(i+1)*b*c], b*c, kr);
        }
        else if (kl != 0 && kl != b*c) {
            shift_left_threads<TYPE>(&a[(i+1)*b*c], b*c, kl);
        }
    }
}

//Performs the first phase of the equidistant gather on chunks of c elements
//r = number of chunks of internal elements (r*c total elements)
//b = number of chunks per leaf node (b*c total elements per leaf node)
//1 grid on the whole array ==> 1 block per cycle
template<typename TYPE>
__global__ void equidistant_gather_chunks_incomplete_phaseOne(TYPE *A, uint64_t r, uint64_t b, uint64_t c) {
    for (uint64_t i = blockIdx.x; i < r; i += gridDim.x) {            //1 block per cycle
        for (uint64_t t = threadIdx.x; t < c; t += blockDim.x) {
            TYPE temp1, temp2;

            temp1 = A[i*c + t];
            A[i*c + t] = A[((i+1)*(b+1) - 1)*c + t];
            temp2 = temp1;

            for (uint64_t j = (i+b)*c; j < ((i+1)*(b+1) - 1)*c; j += b*c) {
                temp1 = A[j + t];
                A[j + t] = temp2;
                temp2 = temp1;
            }
            A[((i+1)*(b+1) - 1)*c + t] = temp2;
        }
    }
}

//Performs the second phase of the equidistant gather on chunks of c elements
//r = number of chunks of internal elements (r*c total elements)
//b = number of chunks per leaf node (b*c total elements per leaf node)
//1 grid on the whole array ==> 1 block per shift
template<typename TYPE>
__global__ void equidistant_gather_chunks_incomplete_phaseTwo(TYPE *A, uint64_t r, uint64_t b, uint64_t c) {
    for (uint64_t i = blockIdx.x; i < r; i += gridDim.x) {      //1 block per shift
        uint64_t kr = (r - i)*c;
        uint64_t kl = (b - r + i)*c;

        if (kr <= kl && kr != 0 && kr != b*c) {
            shift_right_blocks<TYPE>(&A[r*c + i*b*c], b*c, kr);
        }
        else if (kl != 0 && kl != b*c) {
            shift_left_blocks<TYPE>(&A[r*c + i*b*c], b*c, kl);
        }
    }
}

//Performs the equidistant gather on chunks of c elements
//r = number of chunks of internal elements (r*c total elements)
//b = number of chunks per leaf node (b*c total elements per leaf node)
//1 block on the whole array ==> 1 warp per cycle/shift
template<typename TYPE>
__global__ void equidistant_gather_chunks_incomplete_blocks(TYPE *A, uint64_t r, uint64_t b, uint64_t c) {
    int wid = threadIdx.x / WARPS;          //warp id, within block
    int tid = threadIdx.x % WARPS;          //thread id, within warp
    int num_warps = blockDim.x / WARPS;     //total number of warps per block

    for (uint64_t i = wid; i < r; i += num_warps) {            //1 warp per cycle
        for (uint64_t t = tid; t < c; t += WARPS) {
            TYPE temp1, temp2;

            temp1 = A[i*c + t];
            A[i*c + t] = A[((i+1)*(b+1) - 1)*c + t];
            temp2 = temp1;

            for (uint64_t j = (i+b)*c; j < ((i+1)*(b+1) - 1)*c; j += b*c) {
                temp1 = A[j + t];
                A[j + t] = temp2;
                temp2 = temp1;
            }
            A[((i+1)*(b+1) - 1)*c + t] = temp2;
        }
    }
    __syncthreads();

    for (uint64_t i = wid; i < r; i += num_warps) {      //1 warp per shift
        uint64_t kr = (r - i)*c;
        uint64_t kl = (b - r + i)*c;

        if (kr <= kl && kr != 0 && kr != b*c) {
            shift_right_warps<TYPE>(&A[r*c + i*b*c], b*c, kr);
        }
        else if (kl != 0 && kl != b*c) {
            shift_left_warps<TYPE>(&A[r*c + i*b*c], b*c, kl);
        }
    }
}


//Performs the equidistant gather on chunks of c elements
//r = number of chunks of internal elements (r*c total elements)
//b = number of chunks per leaf node (b*c total elements per leaf node)
//1 warp on the whole array ==> 1 thread per cycle/shift
template<typename TYPE>
__global__ void equidistant_gather_chunks_incomplete_warps(TYPE *A, uint64_t r, uint64_t b, uint64_t c) {
    for (uint64_t i = threadIdx.x; i < r; i += WARPS) {            //1 thread per cycle
        for (uint64_t t = 0; t < c; ++t) {
            TYPE temp1, temp2;

            temp1 = A[i*c + t];
            A[i*c + t] = A[((i+1)*(b+1) - 1)*c + t];
            temp2 = temp1;

            for (uint64_t j = (i+b)*c; j < ((i+1)*(b+1) - 1)*c; j += b*c) {
                temp1 = A[j + t];
                A[j + t] = temp2;
                temp2 = temp1;
            }
            A[((i+1)*(b+1) - 1)*c + t] = temp2;
        }
    }

    for (uint64_t i = threadIdx.x; i < r; i += WARPS) {      //1 thread per shift
        uint64_t kr = (r - i)*c;
        uint64_t kl = (b - r + i)*c;

        if (kr <= kl && kr != 0 && kr != b*c) {
            shift_right_threads<TYPE>(&A[r*c + i*b*c], b*c, kr);
        }
        else if (kl != 0 && kl != b*c) {
            shift_left_threads<TYPE>(&A[r*c + i*b*c], b*c, kl);
        }
    }
}

//Performs the extended equidistant gather for n = k(b+1), where k = n/(b+1)
template<typename TYPE>
void extended_equidistant_gather2(TYPE *dev_A, uint64_t n, uint64_t b) {
    uint64_t c = 1;
    uint64_t m = (b+1)*(b+1);
    uint64_t num_partitions = (n+1)/m;          //number of full partitions
    uint64_t r = (((n+1) % m) / (b+1)) + 1;     //number of internal/root elements in the last incomplete partition (0 if all partitions are full)

    #ifdef DEBUG
    printf("extended_equidistant_gather2: n = %lu; b = %lu; num_partitions = %lu\n", n, b, num_partitions);
    cudaError_t cudaerr;
    #endif

    if (num_partitions == 0) {      //only incomplete partition
        equidistant_gather_chunks_incomplete_warps<TYPE><<<1, WARPS>>>(dev_A, r-1, b, c);
        #ifdef DEBUG
        cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess) {
            printf("equidistant_gather_chunks_incomplete_warps failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
        }
        #else
        cudaDeviceSynchronize();
        #endif

        return;
    }

    //1st round, incomplete partition does not need a shift to gather all internal/root elements in the partition
    #ifdef DEBUG
    printf("(1st round) m = %lu; c = %lu; num_partitions = %lu; r = %lu\n", m, c, num_partitions, r);
    #endif

    //TODO: asynchronous kernel launches
    equidistant_gather_chunks_partitions_warps<<<num_partitions, WARPS>>>(dev_A, n, m, b, c, num_partitions);      //1 warp per block... might not be ideal?
    #ifdef DEBUG
    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        printf("equidistant_gather_chunks_partitions_warps failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
    }
    #endif

    if (r > 1) {
        equidistant_gather_chunks_incomplete_warps<<<1, WARPS>>>(&dev_A[m*num_partitions], r-1, b, c);
        #ifdef DEBUG
        cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess) {
            printf("equidistant_gather_chunks_incomplete_warps failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
        }
        #endif
    }
    #ifndef DEBUG
    cudaDeviceSynchronize();
    #endif

    uint64_t r_c = num_partitions % (b+1);
    if (num_partitions < b+1) --r_c;        //if less than b+1 partitions, then internal elements in the first partition do not need to be permuted
    num_partitions /= (b+1);
    m *= (b+1);
    c *= (b+1);

    //Warp level rounds
    while (num_partitions >= 1 && c <= WARPS) {
        #ifdef DEBUG
        printf("(WARPS) m = %lu; c = %lu; num_partitions = %lu; r_c = %lu; r = %lu\n", m, c, num_partitions, r_c, r);
        #endif

        //TODO: asynchronous kernel launches
        equidistant_gather_chunks_partitions_warps<TYPE><<<num_partitions, WARPS>>>(dev_A, n, m, b, c, num_partitions);      //1 warp per block... might not be ideal?
        #ifdef DEBUG
        cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess) {
            printf("equidistant_gather_chunks_partitions_warps failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
        }
        #endif

        if (r_c > 1) {
            equidistant_gather_chunks_incomplete_warps<TYPE><<<1, WARPS>>>(&dev_A[num_partitions*m - 1 + c], r_c-1, b, c);
            #ifdef DEBUG
            cudaerr = cudaDeviceSynchronize();
            if (cudaerr != cudaSuccess) {
                printf("equidistant_gather_chunks_incomplete_warps failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
            }
            #endif
        }
        if (r_c > 0) {           
            shift_right_phaseOne<TYPE><<<1, WARPS>>>(&dev_A[num_partitions*m - 1 + r_c*c], r_c*b*c + r, r);
            #ifdef DEBUG
            cudaerr = cudaDeviceSynchronize();
            if (cudaerr != cudaSuccess) {
                printf("shift_right_phaseOne failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
            }
            #endif

            shift_right_phaseTwo<TYPE><<<1, WARPS>>>(&dev_A[num_partitions*m - 1 + r_c*c], r_c*b*c + r, r);
            #ifdef DEBUG
            cudaerr = cudaDeviceSynchronize();
            if (cudaerr != cudaSuccess) {
                printf("shift_right_phaseTwo failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
            }
            #endif
        }
        #ifdef DEBUG
        cudaDeviceSynchronize();
        #endif

        r += r_c * c;
        r_c = num_partitions % (b+1);
        if (num_partitions < b+1) --r_c;        //if less than b+1 partitions, then internal elements in the first partition do not need to be permuted
        num_partitions /= (b+1);
        m *= (b+1);
        c *= (b+1);
    }

    //Block level rounds
    while (num_partitions >= 1 && c <= THREADS) {
        #ifdef DEBUG
        printf("(BLOCKS) m = %lu; c = %lu; num_partitions = %lu; r_c = %lu; r = %lu\n", m, c, num_partitions, r_c, r);
        #endif

        //TODO: asynchronous kernel launches
        equidistant_gather_chunks_partitions_blocks<TYPE><<<num_partitions, THREADS>>>(dev_A, m, b, c, num_partitions);
        #ifdef DEBUG
        cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess) {
            printf("equidistant_gather_chunks_partitions_blocks failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
        }
        #endif

        if (r_c > 1) {
            equidistant_gather_chunks_incomplete_blocks<TYPE><<<1, THREADS>>>(&dev_A[num_partitions*m - 1 + c], r_c-1, b, c);
            #ifdef DEBUG
            cudaerr = cudaDeviceSynchronize();
            if (cudaerr != cudaSuccess) {
                printf("equidistant_gather_chunks_incomplete_blocks failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
            }
            #endif
        }
        if (r_c > 0) {           
            shift_right_phaseOne<TYPE><<<1, THREADS>>>(&dev_A[num_partitions*m - 1 + r_c*c], r_c*b*c + r, r);
            #ifdef DEBUG
            cudaerr = cudaDeviceSynchronize();
            if (cudaerr != cudaSuccess) {
                printf("shift_right_phaseOne failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
            }
            #endif

            shift_right_phaseTwo<TYPE><<<1, THREADS>>>(&dev_A[num_partitions*m - 1 + r_c*c], r_c*b*c + r, r);
            #ifdef DEBUG
            cudaerr = cudaDeviceSynchronize();
            if (cudaerr != cudaSuccess) {
                printf("shift_right_phaseTwo failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
            }
            #endif
        }
        #ifndef DEBUG
        cudaDeviceSynchronize();
        #endif

        r += r_c * c;
        r_c = num_partitions % (b+1);
        if (num_partitions < b+1) --r_c;        //if less than b+1 partitions, then internal elements in the first partition do not need to be permuted
        num_partitions /= (b+1);
        m *= (b+1);
        c *= (b+1);
    }

    //Grid level rounds
    while (num_partitions >= 1) {
        #ifdef DEBUG
        printf("(GRID) m = %lu; c = %lu; num_partitions = %lu; r_c = %lu; r = %lu\n", m, c, num_partitions, r_c, r);
        #endif

        //TODO: asynchronous kernel launches
        int blocks = num_partitions * (c / THREADS);
        equidistant_gather_chunks_partitions_phaseOne<TYPE><<<blocks, THREADS>>>(dev_A, m, b, c, num_partitions);
        #ifdef DEBUG
        cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess) {
            printf("equidistant_gather_chunks_partitions_phaseOne failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
        }
        #endif

        equidistant_gather_chunks_partitions_phaseTwo<TYPE><<<blocks, THREADS>>>(dev_A, m, b, c, num_partitions);
        #ifdef DEBUG
        cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess) {
            printf("equidistant_gather_chunks_partitions_phaseTwo failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
        }
        #endif

        if (r_c > 1) {
            equidistant_gather_chunks_incomplete_phaseOne<TYPE><<<r_c, THREADS>>>(&dev_A[num_partitions*m - 1 + c], r_c-1, b, c);
            #ifdef DEBUG
	        cudaerr = cudaDeviceSynchronize();
	        if (cudaerr != cudaSuccess) {
	            printf("equidistant_gather_chunks_incomplete_phaseOne failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
	        }
	        #endif

            equidistant_gather_chunks_incomplete_phaseTwo<TYPE><<<r_c, THREADS>>>(&dev_A[num_partitions*m - 1 + c], r_c-1, b, c);
            #ifdef DEBUG
	        cudaerr = cudaDeviceSynchronize();
	        if (cudaerr != cudaSuccess) {
	            printf("equidistant_gather_chunks_incomplete_phaseTwo failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
	        }
	        #endif
        }
        if (r_c > 0) {           
            shift_right_phaseOne<TYPE><<<BLOCKS, THREADS>>>(&dev_A[num_partitions*m - 1 + r_c*c], r_c*b*c + r, r);
            #ifdef DEBUG
	        cudaerr = cudaDeviceSynchronize();
	        if (cudaerr != cudaSuccess) {
	            printf("shift_right_phaseOne failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
	        }
	        #endif

            shift_right_phaseTwo<TYPE><<<BLOCKS, THREADS>>>(&dev_A[num_partitions*m - 1 + r_c*c], r_c*b*c + r, r);
            #ifdef DEBUG
	        cudaerr = cudaDeviceSynchronize();
	        if (cudaerr != cudaSuccess) {
	            printf("shift_right_phaseTwo failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
	        }
	        #endif
        }
        #ifndef DEBUG
        cudaDeviceSynchronize();
        #endif

        r += r_c * c;
        r_c = num_partitions % (b+1);
        if (num_partitions < b+1) --r_c;        //if less than b+1 partitions, then internal elements in the first partition do not need to be permuted
        num_partitions /= (b+1);
        m *= (b+1);
        c *= (b+1);
    }

    //Last round, only incomplete partition remains
    #ifdef DEBUG
    printf("(Last round) m = %lu; c = %lu; r_c = %lu; r = %lu\n", m, c, r_c, r);
    #endif
    if (r_c > 0) {
        /*if (c <= WARPS) {
            equidistant_gather_chunks_incomplete_warps<TYPE><<<1, WARPS>>>(&dev_A[c-1], r_c, b, c);
        }
        else */if (c <= THREADS) {
            equidistant_gather_chunks_incomplete_blocks<TYPE><<<1, THREADS>>>(&dev_A[c-1], r_c, b, c);
            #ifdef DEBUG
            cudaerr = cudaDeviceSynchronize();
            if (cudaerr != cudaSuccess) {
                printf("equidistant_gather_chunks_incomplete_blocks failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
            }
            #endif
        }
        else {
            equidistant_gather_chunks_incomplete_phaseOne<TYPE><<<b, THREADS>>>(&dev_A[c-1], r_c, b, c);
            #ifdef DEBUG
            cudaerr = cudaDeviceSynchronize();
            if (cudaerr != cudaSuccess) {
                printf("equidistant_gather_chunks_incomplete_phaseOne failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
            }
            #endif
            equidistant_gather_chunks_incomplete_phaseTwo<TYPE><<<b, THREADS>>>(&dev_A[c-1], r_c, b, c);
            #ifdef DEBUG
            cudaerr = cudaDeviceSynchronize();
            if (cudaerr != cudaSuccess) {
                printf("equidistant_gather_chunks_incomplete_phaseTwo failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
            }
            #endif
        }
    }

    shift_right_phaseOne<TYPE><<<BLOCKS, THREADS>>>(&dev_A[c-1 + r_c*c], (r_c+1)*b*c + r, r);
    #ifdef DEBUG
    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        printf("shift_right_phaseOne failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
    }
    #endif

    shift_right_phaseTwo<TYPE><<<BLOCKS, THREADS>>>(&dev_A[c-1 + r_c*c], (r_c+1)*b*c + r, r);
    #ifdef DEBUG
    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        printf("shift_right_phaseTwo failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
    }
    #else
    cudaDeviceSynchronize();
    #endif
}

//Performs the extended equidistant gather for n = (b+1)^d - 1, where d is an arbitrary integer
template<typename TYPE>
void extended_equidistant_gather(TYPE *dev_A, uint64_t n, uint64_t b, uint32_t d) {
    if (d == 1) return;

    uint64_t c = 1;
    uint64_t m = (b+1)*(b+1);           //pow(b+1, 2);
    uint64_t num_partitions = (n+1)/m;

    #ifdef DEBUG
    printf("\nextended_equidistant_gather: n = %lu; b = %lu; d = %u; num_partitions = %lu\n", n, b, d, num_partitions);
    struct timespec start, end;
    double ms;
    cudaError_t cudaerr;
    #endif

    //Warp level: 1 warp per partition
    while (num_partitions >= 1 && c <= WARPS) {
        #ifdef DEBUG
        printf("(WARP) m = %lu; b = %lu; c = %lu; num_partitions = %lu\n", m, b, c, num_partitions);
        clock_gettime(CLOCK_MONOTONIC, &start);
        #endif

        //Using 1 warp per block (suprisingly) performs faster than multiple warps per block
        equidistant_gather_chunks_partitions_warps<TYPE><<<num_partitions, WARPS>>>(dev_A, n, m, b, c, num_partitions);
        //int blocks = (num_partitions + (THREADS/WARPS) - 1) / (THREADS/WARPS);      //ceiling(num_partitions/(THREADS/WARPS))
        //equidistant_gather_chunks_partitions_warps<TYPE><<<blocks, THREADS>>>(dev_A, n, m, b, c, num_partitions);

        #ifdef DEBUG
        cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess) {
            printf("equidistant_gather_chunks_partitions_warps failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        ms = ((end.tv_sec*1000000000. + end.tv_nsec) - (start.tv_sec*1000000000. + start.tv_nsec)) / 1000000.;   //millisecond
        printf("\t==> %f ms\n", ms);
        #else
        cudaDeviceSynchronize();
        #endif

        m *= (b+1);
        num_partitions /= (b+1);
        c *= (b+1);
    }

    //Block level
    while (num_partitions >= 1 && c <= THREADS) {
        #ifdef DEBUG
        printf("(BLOCK) m = %lu; b = %lu; c = %lu; num_partitions = %lu\n", m, b, c, num_partitions);
        clock_gettime(CLOCK_MONOTONIC, &start);
        #endif

        equidistant_gather_chunks_partitions_blocks<TYPE><<<num_partitions, THREADS>>>(dev_A, m, b, c, num_partitions);
        #ifdef DEBUG
        cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess) {
            printf("equidistant_gather_chunks_partitions_blocks failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        ms = ((end.tv_sec*1000000000. + end.tv_nsec) - (start.tv_sec*1000000000. + start.tv_nsec)) / 1000000.;   //millisecond
        printf("\t==> %f ms\n", ms);
        #else
        cudaDeviceSynchronize();
        #endif

        m *= (b+1);
        num_partitions /= (b+1);
        c *= (b+1);
    }

    //Grid level
    while (num_partitions >= 1) {
        #ifdef DEBUG
        printf("(GRID) m = %lu; b = %lu; c = %lu; num_partitions = %lu\n", m, b, c, num_partitions);
        clock_gettime(CLOCK_MONOTONIC, &start);
        #endif

        int blocks = num_partitions * (c / THREADS);
        if (blocks == 0) blocks = 1;

        equidistant_gather_chunks_partitions_phaseOne<TYPE><<<blocks, THREADS>>>(dev_A, m, b, c, num_partitions);
        #ifdef DEBUG
        cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess) {
            printf("equidistant_gather_chunks_partitions_phaseTwo failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
        }
        #endif

        equidistant_gather_chunks_partitions_phaseTwo<TYPE><<<blocks, THREADS>>>(dev_A, m, b, c, num_partitions);
        #ifdef DEBUG
        cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess) {
            printf("equidistant_gather_chunks_partitions_phaseTwo failed with error %i \"%s\".\n", cudaerr, cudaGetErrorString(cudaerr));
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        ms = ((end.tv_sec*1000000000. + end.tv_nsec) - (start.tv_sec*1000000000. + start.tv_nsec)) / 1000000.;   //millisecond
        printf("\t==> %f ms\n", ms);
        #else
        cudaDeviceSynchronize();
        #endif

        m *= (b+1);
        num_partitions /= (b+1);
        c *= (b+1);
    }
}

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