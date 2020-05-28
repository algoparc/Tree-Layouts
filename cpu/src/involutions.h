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

#ifndef INVOLUTIONS_H
#define INVOLUTIONS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>

#include "params.h"

uint64_t reverse(uint64_t x);
uint64_t rev_d(uint64_t i, uint64_t d);
uint64_t rev_b(uint64_t i, uint64_t d, uint64_t b);
uint64_t rev_base_d(uint64_t x, uint64_t base, uint64_t d);
uint64_t rev_base_b(uint64_t x, uint64_t base, uint64_t d, uint64_t b);

uint64_t gcd(uint64_t a, uint64_t b);
uint64_t egcd(uint64_t a, uint64_t b, int64_t *x, int64_t *y);
uint64_t involution(uint64_t r, uint64_t x, uint64_t m);

//performs the k-way un-shuffle on array A of size n = k^d - 1
//(note: first element moves position)
template<typename TYPE>
void unshuffle(TYPE *A, uint64_t k, uint64_t n, uint64_t d) {
    uint64_t j;
    TYPE temp;

    //PHASE 1: rev_d
    for (uint64_t i = 0; i < n; ++i) {
        j = rev_base_d(i+1, k, d) - 1;
        
        if (i < j) {
            temp = A[i];
            A[i] = A[j];
            A[j] = temp;
        }
    }

    //PHASE 2: rev_{d-1}
    for (uint64_t i = 0; i < n; ++i) {
        j = rev_base_b(i+1, k, d, d-1) - 1; 

        if (i < j) {
            temp = A[i];
            A[i] = A[j];
            A[j] = temp;
        }
    }
}

//performs the k-way un-shuffle on array A of size n = k^d - 1 using p threads
//(note: first element moves position)
template<typename TYPE>
void unshuffle_parallel(TYPE *A, uint64_t k, uint64_t n, uint64_t d, uint32_t p) {
    uint64_t j;
    TYPE temp;

    //PHASE 1: rev_d
    #pragma omp parallel for shared(A, k, n, d, p) private(j, temp) schedule(guided, B) num_threads(p)
    for (uint64_t i = 0; i < n; ++i) {
        j = rev_base_d(i+1, k, d) - 1;
        
        if (i < j) {
            temp = A[i];
            A[i] = A[j];
            A[j] = temp;
        }
    }

    //PHASE 2: rev_{d-1}
    #pragma omp parallel for shared(A, k, n, d, p) private(j, temp) schedule(guided, B) num_threads(p)
    for (uint64_t i = 0; i < n; ++i) {
        j = rev_base_b(i+1, k, d, d-1) - 1; 

        if (i < j) {
            temp = A[i];
            A[i] = A[j];
            A[j] = temp;
        }
    }
}

//performs the k way shuffle on array A of size n = dk (for some integer d)
//(note: first element does not move positions)
template<typename TYPE>
void shuffle_dk(TYPE *A, uint64_t k, uint64_t n) {
    uint64_t y;
    TYPE temp;

    for (uint64_t x = 1; x < n-1; ++x) {
        y = involution(1, x, n-1);

        if (x < y) {
            temp = A[x];
            A[x] = A[y];
            A[y] = temp;
        }
    }

    for (uint64_t x = 1; x < n-1; ++x) {
        y = involution(k, x, n-1);

        if (x < y) {
            temp = A[x];
            A[x] = A[y];
            A[y] = temp;
        }
    }
}

//performs the k way shuffle on array A of size n = dk (for some integer d)
//(note: first element does not move positions)
template<typename TYPE>
void shuffle_dk_parallel(TYPE *A, uint64_t k, uint64_t n, uint32_t p) {
    uint64_t y;
    TYPE temp;

    #pragma omp parallel for shared(A, k, n, p) private(y, temp) schedule(guided, B) num_threads(p)
    for (uint64_t x = 1; x < n-1; ++x) {
        y = involution(1, x, n-1);

        if (x < y) {
            temp = A[x];
            A[x] = A[y];
            A[y] = temp;
        }
    }

    #pragma omp parallel for shared(A, k, n, p) private(y, temp) schedule(guided, B) num_threads(p)
    for (uint64_t x = 1; x < n-1; ++x) {
        y = involution(k, x, n-1);

        if (x < y) {
            temp = A[x];
            A[x] = A[y];
            A[y] = temp;
        }
    }
}

//performs the k way un-shuffle on array A of size n = dk (for some integer d)
//(note: first element does not move positions)
template<typename TYPE>
void unshuffle_dk(TYPE *A, uint64_t k, uint64_t n) {
    //printf("n = %lu; k = %lu\n", n, k);
    //KYLE: Bug! Segmentation fault if n = 0
    //Bug occurs (i.e., function is called with n = 0) when permute_leaves gets called without a full leaf node
    
    uint64_t y;
    TYPE temp;

    for (uint64_t x = 1; x < n-1; ++x) {
        y = involution(k, x, n-1);

        if (x < y) {
            temp = A[x];
            A[x] = A[y];
            A[y] = temp;
        }
    }

    for (uint64_t x = 1; x < n-1; ++x) {
        y = involution(1, x, n-1);

        if (x < y) {
            temp = A[x];
            A[x] = A[y];
            A[y] = temp;
        }
    }
}

//performs the k way un-shuffle on array A of size n = dk (for some integer d)
//(note: first element does not move positions)
template<typename TYPE>
void unshuffle_dk_parallel(TYPE *A, uint64_t k, uint64_t n, uint32_t p) {
    //Bug: Segmentation fault if n = 0
    //Occurs (i.e., function is called with n = 0) when permute_leaves gets called without a full leaf node
    
    uint64_t y;
    TYPE temp;

    #pragma omp parallel for shared(A, k, n, p) private(y, temp) schedule(guided, B) num_threads(p)
    for (uint64_t x = 1; x < n-1; ++x) {
        y = involution(k, x, n-1);

        if (x < y) {
            temp = A[x];
            A[x] = A[y];
            A[y] = temp;
        }
    }

    #pragma omp parallel for shared(A, k, n, p) private(y, temp) schedule(guided, B) num_threads(p)
    for (uint64_t x = 1; x < n-1; ++x) {
        y = involution(1, x, n-1);

        if (x < y) {
            temp = A[x];
            A[x] = A[y];
            A[y] = temp;
        }
    }
}

//shift n contiguous elements by k to the right via array reversals
template<typename TYPE>
void shift_right(TYPE *A, uint64_t n, uint64_t k) {
    uint64_t i, j;
    TYPE temp;

    //stage 1: reverse whole array
    for (i = 0; i < n/2; ++i) {
        j = n - i - 1;

        temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }

    //stage 2: reverse first k elements & last (n - k) elements
    for (i = 0; i < k/2; ++i) {
        j = k - i - 1;

        temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }

    for (i = k; i < (n + k)/2; ++i) {
        j = n - (i - k) - 1;

        temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }
}

//shift n contiguous elements by k to the right via array reversals using p threads
template<typename TYPE>
void shift_right_parallel(TYPE *A, uint64_t n, uint64_t k, uint32_t p) {
    uint64_t i, j;
    TYPE temp;

    //stage 1: reverse whole array
    #pragma omp parallel for shared(A, n, k) private(i, j, temp) schedule(guided, B) num_threads(p)
    for (i = 0; i < n/2; ++i) {
        j = n - i - 1;

        temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }

    //stage 2: reverse first k elements & last (n - k) elements
    #pragma omp parallel for shared(A, n, k) private(i, j, temp) schedule(guided, B) num_threads(p)
    for (i = 0; i < k/2; ++i) {
        j = k - i - 1;

        temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }

    #pragma omp parallel for shared(A, n, k) private(i, j, temp) schedule(guided, B) num_threads(p)
    for (i = k; i < (n + k)/2; ++i) {
        j = n - (i - k) - 1;

        temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }
}

//shift n contiguous elements by k to the left via array reversals
template<typename TYPE>
void shift_left(TYPE *A, uint64_t n, uint64_t k) {
    uint64_t i, j;
    TYPE temp;

    //stage 1: reverse first k elements & last (n - k) elements
    for (i = 0; i < k/2; ++i) {
        j = k - i - 1;

        temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }

    for (i = k; i < (n + k)/2; ++i) {
        j = n - (i - k) - 1;

        temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }

    //stage 2: reverse whole array
    for (i = 0; i < n/2; ++i) {
        j = n - i - 1;

        temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }
}

//shift n contiguous elements by k to the left via array reversals using p threads
template<typename TYPE>
void shift_left_parallel(TYPE *A, uint64_t n, uint64_t k, uint32_t p) {
    uint64_t i, j;
    TYPE temp;

    //stage 1: reverse first k elements & last (n - k) elements
    #pragma omp parallel for shared(A, n, k) private(i, j, temp) schedule(guided, B) num_threads(p)
    for (i = 0; i < k/2; ++i) {
          j = k - i - 1;

          temp = A[i];
          A[i] = A[j];
          A[j] = temp;
    }

    #pragma omp parallel for shared(A, n, k) private(i, j, temp) schedule(guided, B) num_threads(p)
      for (i = k; i < (n + k)/2; ++i) {
          j = n - (i - k) - 1;

          temp = A[i];
          A[i] = A[j];
          A[j] = temp;
    }

    //stage 2: reverse whole array
    #pragma omp parallel for shared(A, n, k) private(i, j, temp) schedule(guided, B) num_threads(p)
    for (i = 0; i < n/2; ++i) {
        j = n - i - 1;

        temp = A[i];
        A[i] = A[j];
        A[j] = temp;
    }
}
#endif