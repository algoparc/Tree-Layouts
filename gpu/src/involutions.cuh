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

#ifndef INVOLUTIONS_CUH
#define INVOLUTIONS_CUH

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "params.cuh"

__device__ uint64_t rev_d(uint64_t i, uint64_t d);
__device__ uint64_t rev_b(uint64_t i, uint64_t d, uint64_t b);
__device__ uint64_t rev_base_d(uint64_t x, uint64_t base, uint64_t d);
__device__ uint64_t rev_base_b(uint64_t x, uint64_t base, uint64_t d, uint64_t b);
__device__ uint64_t gcd(uint64_t a, uint64_t b);
__device__ uint64_t egcd(uint64_t a, uint64_t b);
__device__ uint64_t involution(uint64_t r, uint64_t x, uint64_t m);

//Performs the first involution of the k-way un-shuffle on array A of size n = k^d - 1
//(Note: first element moves position)
//This is also the second involution of the k-way shuffle of n = k^d - 1 elements
template<typename TYPE>
__global__ void unshuffle_phaseOne(TYPE *A, uint64_t k, uint64_t n, uint64_t d) {
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int num_threads = gridDim.x*blockDim.x;

	uint64_t j;
	TYPE temp;

	//PHASE 1: rev_d
	for (uint64_t i = tid; i < n; i += num_threads) {
        j = rev_base_d(i+1, k, d) - 1;
        
        if (i < j) {
            temp = A[i];
            A[i] = A[j];
            A[j] = temp;
        }
    }
}

//Performs the second involution of the k-way un-shuffle on array A of size n = k^d - 1
//(Note: first element moves position)
//This is also the first involution of the k-way shuffle of n = k^d - 1 elements
template<typename TYPE>
__global__ void unshuffle_phaseTwo(TYPE *A, uint64_t k, uint64_t n, uint64_t d) {
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int num_threads = gridDim.x*blockDim.x;

	uint64_t j;
	TYPE temp;

	//PHASE 2: rev_{d-1}
	for (uint64_t i = tid; i < n; i += num_threads) {
        j = rev_base_b(i+1, k, d, d-1) - 1; 

        if (i < j) {
            temp = A[i];
            A[i] = A[j];
            A[j] = temp;
        }
    }
}

//Performs the first involution of the k-way shuffle on array A of size n = dk (for some integer d)
//(Note: first element does not move positions)
//This is also the second involution of the k-way unshuffle of n = dk elements
template<typename TYPE>
__global__ void shuffle_dk_phaseOne(TYPE *A, uint64_t k, uint64_t n) {
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int num_threads = gridDim.x*blockDim.x;

	uint64_t y;
	TYPE temp;

	for (uint64_t x = tid+1; x < n-1; x += num_threads) {      //start loop with 1. not 0
        y = involution(1, x, n-1);

        if (x < y) {
            temp = A[x];
            A[x] = A[y];
            A[y] = temp;
        }
    }
}

//Performs the second involution of the k-way shuffle on array A of size n = dk (for some integer d)
//(Note: first element does not move positions)
//This is also the first involution of the k-way unshuffle of n = dk elements
template<typename TYPE>
__global__ void shuffle_dk_phaseTwo(TYPE *A, uint64_t k, uint64_t n) {
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int num_threads = gridDim.x*blockDim.x;

	uint64_t y;
	TYPE temp;

	for (uint64_t x = tid+1; x < n-1; x += num_threads) {      //start loop with 1, not 0
        y = involution(k, x, n-1);

        if (x < y) {
            temp = A[x];
            A[x] = A[y];
            A[y] = temp;
        }
    }
}

//Performs the first involution to shift n contiguous elements by k to the right via array reversals
//This is also the second involution to perform a left shift of k elements
template<typename TYPE>
__global__ void shift_right_phaseOne(TYPE *A, uint64_t n, uint64_t k) {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int num_threads = gridDim.x*blockDim.x;

    uint64_t j;
    TYPE temp;

    //Reverse whole array
    for (uint64_t i = tid; i < n/2; i += num_threads) {
        j = n - i - 1;

        temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }
}

//Performs the first involution to shift n contiguous elements by k to the right via array reversals
//This is also the first involution to perform a left shift of k elements
template<typename TYPE>
__global__ void shift_right_phaseTwo(TYPE *A, uint64_t n, uint64_t k) {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int num_threads = gridDim.x*blockDim.x;

    uint64_t j;
    TYPE temp;

    //Reverse first k elements
    for (uint64_t i = tid; i < k/2; i += num_threads) {
        j = k - i - 1;
        
        temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }

    //Reverse last (n - k) elements
    for (uint64_t i = tid + k; i < (n + k)/2; i += num_threads) {
        j = n - (i - k) - 1;

        temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }
}

//Performs the first involution of the k-way unshuffle on each partition of m elements in array A of size n
//Assumes n is a multiple of m = k^d - 1
//Assumes gridDim.x is greater than n/m
//(Note: first element of each partition of m elements moves position)
template<typename TYPE>
__global__ void unshuffle_grid_phaseOne(TYPE *A, uint64_t n, uint64_t k, uint64_t m, uint32_t d) {
    uint32_t blocks = gridDim.x / (n/m);                               //number of blocks per unshuffle on m elements
    int pid = blockIdx.x / blocks;                                     //partition id, i.e., which partition of m elements the current block permutes
    int tid = threadIdx.x + (blockIdx.x - pid*blocks)*blockDim.x;      //thread id within the partition

    if (pid == n/m) return;

    TYPE *a = &A[m*pid];
    uint64_t j;
    TYPE temp;

    //PHASE 1: rev_d
    for (uint64_t i = tid; i < m; i += blocks*blockDim.x) {
        j = rev_base_d(i+1, k, d) - 1;

        if (i < j) {
            temp = a[i];
            a[i] = a[j];
            a[j] = temp;
        }
    }
}

//Performs the second involution of the k-way unshuffle on each partition of m elements in array A of size n
//Assumes n is a multiple of m = k^d - 1
//Assumes gridDim.x is greater than n/m
//(Note: first element of each partition of m elements moves position)
template<typename TYPE>
__global__ void unshuffle_grid_phaseTwo(TYPE *A, uint64_t n, uint64_t k, uint64_t m, uint32_t d) {
    uint32_t blocks = gridDim.x / (n/m);                               //number of blocks per unshuffle on m elements
    int pid = blockIdx.x / blocks;                                     //partition id, i.e., which partition of m elements the current block permutes
    int tid = threadIdx.x + (blockIdx.x - pid*blocks)*blockDim.x;      //thread id within the partition

    if (pid == n/m) return;

    TYPE *a = &A[m*pid];
    uint64_t j;
    TYPE temp;

    //PHASE 2: rev_{d-1}
    for (uint64_t i = tid; i < m; i += blocks*blockDim.x) {
        j = rev_base_b(i+1, k, d, d-1) - 1;
        
        if (i < j) {
            temp = a[i];
            a[i] = a[j];
            a[j] = temp;
        }
    }
}

//Performs the first involution of the k-way shuffle on each partition of m elements in array A of size n
//Assumes n is a multiple of m = dk, for some integer d
//Assumes gridDim.x is greater than n/m
//(Note: first element of each partition of m elements does not move position)
template<typename TYPE>
__global__ void shuffle_dk_grid_phaseOne(TYPE *A, uint64_t n, uint64_t k, uint64_t m) {
    uint32_t blocks = gridDim.x / (n/m);                               //number of blocks per unshuffle on m elements
    int pid = blockIdx.x / blocks;                                     //partition id, i.e., which partition of m elements the current block permutes
    int tid = threadIdx.x + (blockIdx.x - pid*blocks)*blockDim.x;      //thread id within the partition

    if (pid == n/m) return;

    TYPE *a = &A[m*pid];
    uint64_t y;
    TYPE temp;

    for (uint64_t x = tid+1; x < m-1; x += blocks*blockDim.x) {
        y = involution(1, x, m-1);

        if (x < y) {
            temp = a[x];
            a[x] = a[y];
            a[y] = temp;
        }
    }
}

//Performs the second involution of the k-way shuffle on each partition of m elements in array A of size n
//Assumes n is a multiple of m = dk, for some integer d
//Assumes gridDim.x is greater than n/m
//(Note: first element of each partition of m elements does not move position)
template<typename TYPE>
__global__ void shuffle_dk_grid_phaseTwo(TYPE *A, uint64_t n, uint64_t k, uint64_t m) {
    uint32_t blocks = gridDim.x / (n/m);                               //number of blocks per unshuffle on m elements
    int pid = blockIdx.x / blocks;                                     //partition id, i.e., which partition of m elements the current block permutes
    int tid = threadIdx.x + (blockIdx.x - pid*blocks)*blockDim.x;      //thread id within the partition

    if (pid == n/m) return;

    TYPE *a = &A[m*pid];
    uint64_t y;
    TYPE temp;

    for (uint64_t x = tid+1; x < m-1; x += blocks*blockDim.x) {
        y = involution(k, x, m-1);

        if (x < y) {
            temp = a[x];
            a[x] = a[y];
            a[y] = temp;
        }
    }
}

//Performs the first involution of the k-way shuffle on each partition of m elements in array A of size n
//Skips the first s elements, i.e., only the last (m - s) elements of each partitioned are shuffled
//Assumes n is a multiple of m = dk, for some integer d
//Assumes gridDim.x is greater than n/m
//(Note: first element of each partition of m elements does not move position)
template<typename TYPE>
__global__ void shuffle_dk_skip_grid_phaseOne(TYPE *A, uint64_t n, uint64_t k, uint64_t m, uint64_t s) {
    uint32_t blocks = gridDim.x / (n/m);                               //number of blocks per unshuffle on m elements
    int pid = blockIdx.x / blocks;                                     //partition id, i.e., which partition of m elements the current block permutes
    int tid = threadIdx.x + (blockIdx.x - pid*blocks)*blockDim.x;      //thread id within the partition

    if (pid == n/m) return;

    TYPE *a = &A[m*pid + s];
    uint64_t y;
    TYPE temp;

    for (uint64_t x = tid+1; x < m-s-1; x += blocks*blockDim.x) {
        y = involution(1, x, m-s-1);

        if (x < y) {
            temp = a[x];
            a[x] = a[y];
            a[y] = temp;
        }
    }
}

//Performs the second involution of the k-way shuffle on each partition of m elements in array A of size n
//Skips the first s elements, i.e., only the last (m - s) elements of each partitioned are shuffled
//Assumes n is a multiple of m = dk, for some integer d
//Assumes gridDim.x is greater than n/m
//(Note: first element of each partition of m elements does not move position)
template<typename TYPE>
__global__ void shuffle_dk_skip_grid_phaseTwo(TYPE *A, uint64_t n, uint64_t k, uint64_t m, uint64_t s) {
    uint32_t blocks = gridDim.x / (n/m);                               //number of blocks per unshuffle on m elements
    int pid = blockIdx.x / blocks;                                     //partition id, i.e., which partition of m elements the current block permutes
    int tid = threadIdx.x + (blockIdx.x - pid*blocks)*blockDim.x;      //thread id within the partition

    if (pid == n/m) return;
  
    TYPE *a = &A[m*pid + s];
    uint64_t y;
    TYPE temp;

    for (uint64_t x = tid+1; x < m-s-1; x += blocks*blockDim.x) {
        y = involution(k, x, m-s-1);

        if (x < y) {
            temp = a[x];
            a[x] = a[y];
            a[y] = temp;
        }
    }
}

//Performs the k-way unshuffle on each partition of m elements in array A of size n
//1 block working on the entire array
//Assumes n is a multiple of m = k^d - 1
//Assumes number of warps = blockDim.x/WARPS is greater than n/m
//(Note: first element of each partition of m elements moves position)
template<typename TYPE>
__device__ void unshuffle_blocks(TYPE *A, uint64_t n, uint64_t k, uint64_t m, uint32_t d) {
    int warps = (blockDim.x/WARPS) / (n/m);                                         //number of warps per unshuffle on m elements
    int pid = (threadIdx.x/WARPS) / warps;                                          //partition id, i.e., which partition of m elements the current warp permutes
    int tid = ((threadIdx.x/WARPS) - pid*warps)*WARPS + (threadIdx.x % WARPS);      //thread id within the partition

    if (pid == n/m) return;

    TYPE *a = &A[m*pid];
    uint64_t j;
    TYPE temp;

    //PHASE 1: rev_d
    for (uint64_t i = tid; i < m; i += warps*WARPS) {
        j = rev_base_d(i+1, k, d) - 1;

        if (i < j) {
            temp = a[i];
            a[i] = a[j];
            a[j] = temp;
        }
    }
    __syncthreads();

    //PHASE 2: rev_{d-1}
    for (uint64_t i = tid; i < m; i += warps*WARPS) {
        j = rev_base_b(i+1, k, d, d-1) - 1;
        
        if (i < j) {
            temp = a[i];
            a[i] = a[j];
            a[j] = temp;
        }
    }
    __syncthreads();
}

//Performs the the k-way shuffle on each partition of m elements in array A of size n
//1 block working on the entire array
//Assumes n is a multiple of m = dk, for some integer d
//Assumes number of warps = blockDim.x/WARPS is greater than n/m
//(Note: first element of each partition of m elements does not move position)
template<typename TYPE>
__device__ void shuffle_dk_blocks(TYPE *A, uint64_t n, uint64_t k, uint64_t m) {
    int warps = (blockDim.x/WARPS) / (n/m);                                         //number of warps per unshuffle on m elements
    int pid = (threadIdx.x/WARPS) / warps;                                          //partition id, i.e., which partition of m elements the current warp permutes
    int tid = ((threadIdx.x/WARPS) - pid*warps)*WARPS + (threadIdx.x % WARPS);      //thread id within the partition

    if (pid == n/m) return;

    TYPE *a = &A[m*pid];
    uint64_t y;
    TYPE temp;

    for (uint64_t x = tid+1; x < m-1; x += warps*WARPS) {
        y = involution(1, x, m-1);

        if (x < y) {
            temp = a[x];
            a[x] = a[y];
            a[y] = temp;
        }
    }
    __syncthreads();

    for (uint64_t x = tid+1; x < m-1; x += warps*WARPS) {
        y = involution(k, x, m-1);

        if (x < y) {
            temp = a[x];
            a[x] = a[y];
            a[y] = temp;
        }
    }
    __syncthreads();
}

//Performs the the k-way unshuffle on each partition of m elements in array A of size n
//1 block working on the entire array
//Assumes n is a multiple of m = dk, for some integer d
//Assumes number of warps = blockDim.x/WARPS is greater than n/m
//(Note: first element of each partition of m elements does not move position)
template<typename TYPE>
__device__ void unshuffle_dk_blocks(TYPE *A, uint64_t n, uint64_t k, uint64_t m) {
    int warps = (blockDim.x/WARPS) / (n/m);                                         //number of warps per unshuffle on m elements
    int pid = (threadIdx.x/WARPS) / warps;                                          //partition id, i.e., which partition of m elements the current warp permutes
    int tid = ((threadIdx.x/WARPS) - pid*warps)*WARPS + (threadIdx.x % WARPS);      //thread id within the partition

    if (pid == n/m) return;

    TYPE *a = &A[m*pid];
    uint64_t y;
    TYPE temp;

    for (uint64_t x = tid+1; x < m-1; x += warps*WARPS) {
        y = involution(k, x, m-1);

        if (x < y) {
            temp = a[x];
            a[x] = a[y];
            a[y] = temp;
        }
    }
    __syncthreads();

    for (uint64_t x = tid+1; x < m-1; x += warps*WARPS) {
        y = involution(1, x, m-1);

        if (x < y) {
            temp = a[x];
            a[x] = a[y];
            a[y] = temp;
        }
    }
    __syncthreads();
}

//Performs the the k-way shuffle on each partition of m elements in array A of size n
//1 block working on the entire array
//Skips the first s elements, i.e., only the last (m - s) elements of each partitioned are shuffled
//Assumes n is a multiple of m = dk, for some integer d
//Assumes number of warps = blockDim.x/WARPS is greater than n/m
//(Note: first element of each partition of m elements does not move position)
template<typename TYPE>
__device__ void shuffle_dk_skip_blocks(TYPE *A, uint64_t n, uint64_t k, uint64_t m, uint64_t s) {
    int warps = (blockDim.x/WARPS) / (n/m);                                         //number of warps per unshuffle on m elements
    int pid = (threadIdx.x/WARPS) / warps;                                          //partition id, i.e., which partition of m elements the current warp permutes
    int tid = ((threadIdx.x/WARPS) - pid*warps)*WARPS + (threadIdx.x % WARPS);      //thread id within the partition

    if (pid == n/m) return;

    TYPE *a = &A[m*pid + s];
    uint64_t y;
    TYPE temp;

    for (uint64_t x = tid+1; x < m-s-1; x += warps*WARPS) {
        y = involution(1, x, m-s-1);

        if (x < y) {
            temp = a[x];
            a[x] = a[y];
            a[y] = temp;
        }
    }
    __syncthreads();

    for (uint64_t x = tid+1; x < m-s-1; x += warps*WARPS) {
        y = involution(k, x, m-s-1);

        if (x < y) {
            temp = a[x];
            a[x] = a[y];
            a[y] = temp;
        }
    }
    __syncthreads();
}

//Performs the k-way unshuffle on each partition of m elements in array A of size n
//1 warp working on the entire array
//Assumes n is a multiple of m = k^d - 1
//Assumes number of threads per warp is greater than n/m
//(Note: first element of each partition of m elements moves position)
template<typename TYPE>
__device__ void unshuffle_warps(TYPE *A, uint64_t n, uint64_t k, uint64_t m, uint32_t d) {
    uint32_t threads = WARPS / (n/m);                   //number of threads per unshuffle on m elements
    int pid = (threadIdx.x % WARPS) / threads;          //partition id, i.e., which partition of m elements the current thread permutes
    int tid = (threadIdx.x % WARPS) - pid*threads;      //thread id within the partition

    if (pid == n/m) return;

    TYPE *a = &A[m*pid];
    uint64_t j;
    TYPE temp;

    //PHASE 1: rev_d
    for (uint64_t i = tid; i < m; i += threads) {
        j = rev_base_d(i+1, k, d) - 1;

        if (i < j) {
            temp = a[i];
            a[i] = a[j];
            a[j] = temp;
        }
    }

    //PHASE 2: rev_{d-1}
    for (uint64_t i = tid; i < m; i += threads) {
        j = rev_base_b(i+1, k, d, d-1) - 1;
        
        if (i < j) {
            temp = a[i];
            a[i] = a[j];
            a[j] = temp;
        }
    }
}

//Performs the the k-way shuffle on each partition of m elements in array A of size n
//1 warp working on the entire array
//Assumes n is a multiple of m = dk, for some integer d
//Assumes number of threads per warp is greater than n/m
//(Note: first element of each partition of m elements does not move position)
template<typename TYPE>
__device__ void shuffle_dk_warps(TYPE *A, uint64_t n, uint64_t k, uint64_t m) {
    uint32_t threads = WARPS / (n/m);                   //number of threads per unshuffle on m elements
    int pid = (threadIdx.x % WARPS) / threads;          //partition id, i.e., which partition of m elements the current thread permutes
    int tid = (threadIdx.x % WARPS) - pid*threads;      //thread id within the partition

    if (pid == n/m) return;

    TYPE *a = &A[m*pid];
    uint64_t y;
    TYPE temp;

    for (uint64_t x = tid+1; x < m-1; x += threads) {
        y = involution(1, x, m-1);

        if (x < y) {
            temp = a[x];
            a[x] = a[y];
            a[y] = temp;
        }
    }

    for (uint64_t x = tid+1; x < m-1; x += threads) {
        y = involution(k, x, m-1);

        if (x < y) {
            temp = a[x];
            a[x] = a[y];
            a[y] = temp;
        }
    }
}

//Performs the the k-way unshuffle on each partition of m elements in array A of size n
//1 warp working on the entire array
//Assumes n is a multiple of m = dk, for some integer d
//Assumes number of threads per warp is greater than n/m
//(Note: first element of each partition of m elements does not move position)
template<typename TYPE>
__device__ void unshuffle_dk_warps(TYPE *A, uint64_t n, uint64_t k, uint64_t m) {
    uint32_t threads = WARPS / (n/m);                   //number of threads per unshuffle on m elements
    int pid = (threadIdx.x % WARPS) / threads;          //partition id, i.e., which partition of m elements the current thread permutes
    int tid = (threadIdx.x % WARPS) - pid*threads;      //thread id within the partition

    if (pid == n/m) return;

    TYPE *a = &A[m*pid];
    uint64_t y;
    TYPE temp;

    for (uint64_t x = tid+1; x < m-1; x += threads) {
        y = involution(k, x, m-1);

        if (x < y) {
            temp = a[x];
            a[x] = a[y];
            a[y] = temp;
        }
    }

    for (uint64_t x = tid+1; x < m-1; x += threads) {
        y = involution(1, x, m-1);

        if (x < y) {
            temp = a[x];
            a[x] = a[y];
            a[y] = temp;
        }
    }
}

//Performs the the k-way shuffle on each partition of m elements in array A of size n
//1 warp working on the entire array
//Skips the first s elements, i.e., only the last (m - s) elements of each partitioned are shuffled
//Assumes n is a multiple of m = dk, for some integer d
//Assumes number of threads per warp is greater than n/m
//(Note: first element of each partition of m elements does not move position)
template<typename TYPE>
__device__ void shuffle_dk_skip_warps(TYPE *A, uint64_t n, uint64_t k, uint64_t m, uint64_t s) {
    uint32_t threads = WARPS / (n/m);                   //number of threads per unshuffle on m elements
    int pid = (threadIdx.x % WARPS) / threads;          //partition id, i.e., which partition of m elements the current thread permutes
    int tid = (threadIdx.x % WARPS) - pid*threads;      //thread id within the partition

    if (pid == n/m) return;

    TYPE *a = &A[m*pid + s];
    uint64_t y;
    TYPE temp;

    for (uint64_t x = tid+1; x < m-s-1; x += threads) {
        y = involution(1, x, m-s-1);

        if (x < y) {
            temp = a[x];
            a[x] = a[y];
            a[y] = temp;
        }
    }

    for (uint64_t x = tid+1; x < m-s-1; x += threads) {
        y = involution(k, x, m-s-1);

        if (x < y) {
            temp = a[x];
            a[x] = a[y];
            a[y] = temp;
        }
    }
}
#endif