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

#ifndef CYCLES_CUH
#define CYCLES_CUH

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include <math.h>

#include "params.cuh"

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

//Shifts n contiguous elements by k to the right via array reversals
//1 block on the entire array
template<typename TYPE>
__device__ void shift_right_blocks(TYPE *A, uint64_t n, uint64_t k) {
    uint64_t j;
    TYPE temp;

    //stage 1: reverse whole array
    for (uint64_t i = threadIdx.x; i < n/2; i += blockDim.x) {
        j = n - i - 1;

        temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }
    __syncthreads();

    //stage 2: reverse first k elements & last (n - k) elements
    for (uint64_t i = threadIdx.x; i < k/2; i += blockDim.x) {
        j = k - i - 1;

        temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }

    for (uint64_t i = threadIdx.x + k; i < (n + k)/2; i += blockDim.x) {
        j = n - (i - k) - 1;

        temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }
    __syncthreads();
}

//Shifts n contiguous elements by k to the right via array reversals
//1 warp on the entire array
template<typename TYPE>
__device__ void shift_right_warps(TYPE *A, uint64_t n, uint64_t k) {
    int tid = threadIdx.x % WARPS;      //thread id within the warp
    uint64_t j;
    TYPE temp;

    //stage 1: reverse whole array
    for (uint64_t i = tid; i < n/2; i += WARPS) {
        j = n - i - 1;

        temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }

    //stage 2: reverse first k elements & last (n - k) elements
    for (uint64_t i = tid; i < k/2; i += WARPS) {
        j = k - i - 1;

        temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }

    for (uint64_t i = tid + k; i < (n + k)/2; i += WARPS) {
        j = n - (i - k) - 1;

        temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }
}

//Shifts n contiguous elements by k to the right via array reversals
//1 thread on the entire array
template<typename TYPE>
__device__ void shift_right_threads(TYPE *A, uint64_t n, uint64_t k) {
    uint64_t j;
    TYPE temp;

    //stage 1: reverse whole array
    for (uint64_t i = 0; i < n/2; ++i) {
        j = n - i - 1;

        temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }

    //stage 2: reverse first k elements & last (n - k) elements
    for (uint64_t i = 0; i < k/2; ++i) {
        j = k - i - 1;

        temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }

    for (uint64_t i = k; i < (n + k)/2; ++i) {
        j = n - (i - k) - 1;

        temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }
}

//Shifts n contiguous elements by k to the right via array reversals
//1 block on the entire array
template<typename TYPE>
__device__ void shift_left_blocks(TYPE *A, uint64_t n, uint64_t k) {
    uint64_t j;
    TYPE temp;

    //stage 1: reverse first k elements & last (n - k) elements
    for (uint64_t i = threadIdx.x; i < k/2; i += blockDim.x) {
        j = k - i - 1;

        temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }

    for (uint64_t i = threadIdx.x + k; i < (n + k)/2; i += blockDim.x) {
        j = n - (i - k) - 1;

        temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }
    __syncthreads();

    //stage 2: reverse whole array
    for (uint64_t i = threadIdx.x; i < n/2; i += blockDim.x) {
        j = n - i - 1;

        temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }
    __syncthreads();
}

//Shifts n contiguous elements by k to the right via array reversals
//1 warp on the entire array
template<typename TYPE>
__device__ void shift_left_warps(TYPE *A, uint64_t n, uint64_t k) {
    int tid = threadIdx.x % WARPS;      //thread id within the warp
    uint64_t j;
    TYPE temp;

    //stage 1: reverse first k elements & last (n - k) elements
    for (uint64_t i = tid; i < k/2; i += WARPS) {
        j = k - i - 1;

        temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }

    for (uint64_t i = tid + k; i < (n + k)/2; i += WARPS) {
        j = n - (i - k) - 1;

        temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }

    //stage 2: reverse whole array
    for (uint64_t i = tid; i < n/2; i += WARPS) {
        j = n - i - 1;

        temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }
}

//Shifts n contiguous elements by k to the right via array reversals
//1 thread on the entire array
template<typename TYPE>
__device__ void shift_left_threads(TYPE *A, uint64_t n, uint64_t k) {
    uint64_t j;
    TYPE temp;

    //stage 1: reverse first k elements & last (n - k) elements
    for (uint64_t i = 0; i < k/2; ++i) {
        j = k - i - 1;

        temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }

    for (uint64_t i = k; i < (n + k)/2; ++i) {
        j = n - (i - k) - 1;

        temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }

    //stage 2: reverse whole array
    for (uint64_t i = 0; i < n/2; ++i) {
        j = n - i - 1;

        temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }
}

//Performs the first phase (r cycles) of the equidistant gather on r root elements and l leaf elements
//Assumes r <= l
//1 grid on the entire array
template<typename TYPE>
__global__ void equidistant_gather_phaseOne(TYPE *A, uint64_t r, uint64_t l) {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int num_threads = gridDim.x * blockDim.x;

    for (uint64_t i = tid; i < r; i += num_threads) {
        TYPE temp1, temp2;

        temp1 = A[i];
        A[i] = A[(i+1)*(l+1) - 1];
        temp2 = temp1;

        for (uint64_t j = i+l; j < (i+1)*(l+1) - 1; j += l) {
            temp1 = A[j];
            A[j] = temp2;
            temp2 = temp1;
        }
        A[(i+1)*(l+1) - 1] = temp2;
    }
}

//Performs the second phase (shifts of leaf subtrees) of the equidistant gather on r root elements and l leaf elements
//Assumes r <= l
//1 grid on the entire array
template<typename TYPE>
__global__ void equidistant_gather_phaseTwo(TYPE *A, uint64_t r, uint64_t l) {
    //1 block per shift
    for (uint64_t i = blockIdx.x; i < r; i += gridDim.x) {      //leaf subtrees are 0-indexed
        //right shift of r - i or left shift of l - (r - i)
        //A[r + i*l] to A[r + (i+1)*l - 1]

        uint64_t kr = r - i;
        uint64_t kl = l - r + i;

        if (kr <= kl && kr != 0 && kr != l) {
            shift_right_blocks<TYPE>(&A[r + i*l], l, kr);
        }
        else if (kl != 0 && kl != l) {
            shift_left_blocks<TYPE>(&A[r + i*l], l, kl);
        }
    }
}

//Performs the first phase (r cycles) of the equidistant gather on each partition of m elements
//r root elements and l leaf elements per partition
//Assumes n is a multiple of m
//Assumes r <= l
//Assumes gridDim.x >= n/m
//1 grid on the entire array
template<typename TYPE>
__global__ void equidistant_gather_partitions_phaseOne(TYPE *A, uint64_t n, uint64_t m, uint64_t r, uint64_t l) {
    uint32_t blocks = gridDim.x / (n/m);                                //number of blocks per unshuffle on m elements
    int pid = blockIdx.x / blocks;                                      //partition id, i.e., which partition of m elements the current block permutes
    int tid = threadIdx.x + (blockIdx.x - pid*blocks)*blockDim.x;       //thread id within the partition

    TYPE *a = &A[m*pid];            //pointer to start of partition for the block

    for (uint64_t i = tid; i < r; i += blocks*blockDim.x) {
        TYPE temp1, temp2;

        temp1 = a[i];
        a[i] = a[(i+1)*(l+1) - 1];
        temp2 = temp1;

        for (uint64_t j = i+l; j < (i+1)*(l+1) - 1; j += l) {
            temp1 = a[j];
            a[j] = temp2;
            temp2 = temp1;
        }
        a[(i+1)*(l+1) - 1] = temp2;
    }
}

//Performs the second phase (shifts of leaf subtrees) of the equidistant gather on each partition of m elements
//r root elements and l leaf elements per partition
//Assumes n is a multiple of m
//Assumes r <= l
//Assumes gridDim.x >= n/m
//1 grid on the entire array
template<typename TYPE>
__global__ void equidistant_gather_partitions_phaseTwo(TYPE *A, uint64_t n, uint64_t m, uint64_t r, uint64_t l) {
    uint32_t blocks = gridDim.x / (n/m);                    //number of blocks per unshuffle on m elements
    int pid = blockIdx.x / blocks;                          //partition id, i.e., which partition of m elements the current block permutes
    int bid = blockIdx.x - pid*blocks;                      //block id within the partition

    TYPE *a = &A[m*pid];            //pointer to start of partition for the block

    //1 block per shift
    for (uint64_t i = bid; i < r; i += blocks) {      //leaf subtrees are 0-indexed
        //right shift of r - i or left shift of l - (r - i)
        //a[r + i*l] to a[r + (i+1)*l - 1]

        uint64_t kr = r - i;
        uint64_t kl = l - r + i;

        if (kr <= kl && kr != 0 && kr != l) {
            shift_right_blocks<TYPE>(&a[r + i*l], l, kr);
        }
        else if (kl != 0 && kl != l) {
            shift_left_blocks<TYPE>(&a[r + i*l], l, kl);
        }
    }
}

//Performs the equidistant gather on r root elements and l leaf elements
//Assumes r <= l
//1 block on the entire array
template<typename TYPE>
__device__ void equidistant_gather_blocks(TYPE *A, uint64_t r, uint64_t l) {
    for (uint64_t i = threadIdx.x; i < r; i += blockDim.x) {       //front-to-back (I/O-efficient, i.e., performs coalesced global accesses)
        TYPE temp1, temp2;

        temp1 = A[i];
        A[i] = A[(i+1)*(l+1) - 1];
        temp2 = temp1;

        for (uint64_t j = i+l; j < (i+1)*(l+1) - 1; j += l) {
            temp1 = A[j];
            A[j] = temp2;
            temp2 = temp1;
        }
        A[(i+1)*(l+1) - 1] = temp2;
    }
    __syncthreads();

    int wid = threadIdx.x / WARPS;
    int num_warps = blockDim.x / WARPS;

    //1 warp per shift
    for (uint64_t i = wid; i < r; i += num_warps) {
        //right shift of r - i or left shift of l - (r - i)
        //A[r + i*l] to A[r + (i+1)*l - 1]

        uint64_t kr = r - i;
        uint64_t kl = l - r + i;

        if (kr <= kl && kr != 0 && kr != l) {
            shift_right_warps<TYPE>(&A[r + i*l], l, kr);
        }
        else if (kl != 0 && kl != l) {
            shift_left_warps<TYPE>(&A[r + i*l], l, kl);
        }
    }
    __syncthreads();
}

//Performs the equidistant gather on r root elements and l leaf elements
//Assumes r <= l
//1 warp on the entire array
template<typename TYPE>
__device__ void equidistant_gather_warps(TYPE *A, uint64_t r, uint64_t l) {
    int tid = threadIdx.x % WARPS;      //thread id within warp

    for (uint64_t i = tid; i < r; i += WARPS) {       //front-to-back (I/O-efficient, i.e., performs coalesced global accesses)
        TYPE temp1, temp2;

        temp1 = A[i];
        A[i] = A[(i+1)*(l+1) - 1];
        temp2 = temp1;

        for (uint64_t j = i+l; j < (i+1)*(l+1) - 1; j += l) {
            temp1 = A[j];
            A[j] = temp2;
            temp2 = temp1;
        }
        A[(i+1)*(l+1) - 1] = temp2;
    }

    //1 thread per shift
    for (uint64_t i = tid; i < r; i += WARPS) {
        if (r - i != 0 && r - i != l) {
            //perform a shift right by (r - i) elements on the i-th leaf subtree
            //we want threads to perform the same function so that they do not diverge, hence, we always perform a right shift regardless of the shift value
            shift_right_threads<TYPE>(&A[r + i*l], l, r - i);         
        }
    }
}
#endif