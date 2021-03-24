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

#ifndef CYCLES_CUH
#define CYCLES_CUH

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include <math.h>

#include "params.cuh"

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
    uint64_t num_partitions = n/m;                                      //number of partitions in array
    uint64_t blocks = gridDim.x / num_partitions;                       //number of blocks per unshuffle on m elements
    int pid = blockIdx.x / blocks;                                      //partition id, i.e., which partition of m elements the current block permutes
    int tid = threadIdx.x + (blockIdx.x - pid*blocks)*blockDim.x;       //thread id within the partition

    TYPE *a = &A[m*pid];            //pointer to start of partition for the block

    if (pid < num_partitions) {
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
}

//Performs the second phase (shifts of leaf subtrees) of the equidistant gather on each partition of m elements
//r root elements and l leaf elements per partition
//Assumes n is a multiple of m
//Assumes r <= l
//Assumes gridDim.x >= n/m
//1 grid on the entire array
template<typename TYPE>
__global__ void equidistant_gather_partitions_phaseTwo(TYPE *A, uint64_t n, uint64_t m, uint64_t r, uint64_t l) {
    uint64_t num_partitions = n/m;                          //number of partitions in array
    uint64_t blocks = gridDim.x / num_partitions;           //number of blocks per unshuffle on m elements
    int pid = blockIdx.x / blocks;                          //partition id, i.e., which partition of m elements the current block permutes
    int bid = blockIdx.x - pid*blocks;                      //block id within the partition

    TYPE *a = &A[m*pid];            //pointer to start of partition for the block

    if (pid < num_partitions) {
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
//Assumes 1 warp per thread block
//1 warp per partition ==> 1 thread per cycle/shift
template<typename TYPE>
__global__ void equidistant_gather_chunks_partitions_warps(TYPE *A, uint64_t n, uint64_t m, uint64_t b, uint64_t c, uint32_t num_partitions) {
    //int wid = (threadIdx.x + blockIdx.x*blockDim.x)/WARPS;      //global warp id
    int wid = blockIdx.x;                   //global warp id (assumes 1 warp per block)
    int tid = threadIdx.x % WARPS;          //thread id within warp

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
#endif