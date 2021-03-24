/*
 * Copyright 2018-2021 Kyle Berney
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

#ifndef VEB_INVOLUTIONS_H
#define VEB_INVOLUTIONS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#include "common.h"
#include "involutions.h"

//Permutes sorted array into the van Emde Boas tree layout via Level-order B-tree involutions
//B = # of leaf elements per leaf subtree, i.e., paramter l (in the below code)
//The top subtree has a height of floor{(h - 1)/2} and leaf subtrees have height of ceil{(h - 1)/2}
template<typename TYPE>
void permutevEB(TYPE *A, uint64_t n, uint32_t d) {
	if (n == 1) return;

    uint32_t root_d = (d - 2)/2 + 1;		//floor((d - 2)/2) + 1
    uint32_t leaf_d = d - root_d;			//ceil((d - 2)/2.) + 1

    uint64_t r = pow(2, root_d) - 1;		//number of elements in root subtree
    uint64_t l = pow(2, leaf_d) - 1;		//number of elements in leaf subtrees

    uint32_t h = log10(n)/log10(l+1);
    if (n != pow(l+1, h+1) - 1) {
        //printf("non-perfect B-tree of height %ld\n", h);
    	unshuffle_dk<TYPE>(A - 1, l+1, n + 1);      
        shuffle_dk<TYPE>(&A[r], l, n - r);
    }
    else {
        //printf("perfect B-tree of height %ld\n", h);
        unshuffle<TYPE>(A, l+1, n, h+1);
        shuffle_dk<TYPE>(&A[n/(l+1)], l, pow(l+1, h) * l);
    }
    
    permutevEB<TYPE>(A, r, root_d);               //recurse on root subtree

    uint32_t numLeafTrees = (n - r)/l;
    for (int i = 0; i < numLeafTrees; i++) {
        permutevEB<TYPE>(&A[r + i*l], l, leaf_d);        //recurse on i-th leaf subtree
    }
}

//Permutes sorted array into the van Emde Boas tree layout via Level-order B-tree involutions using p processors
//B = # of leaf elements per leaf subtree, i.e., paramter l (in the below code)
//The top subtree has a height of ceil{(h - 1)/2} and leaf subtrees have height of floor{(h - 1)/2}
template<typename TYPE>
void permutevEB_parallel(TYPE *A, uint64_t n, uint32_t d, uint32_t p) {
    if (n == 1) return;

    uint32_t root_d = (d - 2)/2 + 1;        //floor((d - 2)/2) + 1
    uint32_t leaf_d = d - root_d;           //ceil((d - 2)/2.) + 1

    uint64_t r = pow(2, root_d) - 1;        //number of elements in root subtree
    uint64_t l = pow(2, leaf_d) - 1;        //number of elements in leaf subtrees

    uint32_t h = log10(n)/log10(l+1);
    if (n != pow(l+1, h+1) - 1) {
        //printf("non-perfect B-tree of height %ld\n", h);
        unshuffle_dk_parallel<TYPE>(A - 1, l+1, n + 1, p);
        shuffle_dk_parallel<TYPE>(&A[r], l, n - r, p);
    }
    else {
        //printf("perfect B-tree of height %ld\n", h);
        unshuffle_parallel<TYPE>(A, l+1, n, h+1, p);
        shuffle_dk_parallel<TYPE>(&A[n/(l+1)], l, pow(l+1, h) * l, p);
    }
    
    //Recurse
    if (root_d == leaf_d) {
        if (p <= n/r) {
            #pragma omp parallel for shared(A, n, d, p, root_d, leaf_d, r, l, h) schedule(guided) num_threads(p)
            for (uint64_t i = 0; i < n; i += r) {
                permutevEB<TYPE>(A, r, root_d);
            }
        }
        else {
            uint32_t threads_per = ceil(p/(double)(n/r));
            #pragma omp parallel for shared(A, n, d, p, root_d, leaf_d, r, l, h, threads_per) schedule(guided) num_threads(n/r)
            for (uint64_t i = 0; i < n; i += r) {
                permutevEB_parallel<TYPE>(A, r, root_d, threads_per);
            }
        }
    }
    else {
        permutevEB_parallel<TYPE>(A, r, root_d, p);

        uint32_t numLeafTrees = (n-r)/l;
        if (p <= numLeafTrees) {
            #pragma omp parallel for shared(A, n, d, p, root_d, leaf_d, r, l, h, numLeafTrees) schedule(guided) num_threads(p)
            for (uint64_t i = r; i < n; i += l) {
                permutevEB<TYPE>(A, l, leaf_d);
            }
        }
        else {
            uint32_t threads_per = ceil(p/(double)numLeafTrees);

            #pragma omp parallel for shared(A, n, d, p, root_d, leaf_d, r, l, h, numLeafTrees, threads_per) schedule(guided) num_threads(numLeafTrees)
            for (uint64_t i = r; i < n; i += l) {
                permutevEB_parallel<TYPE>(A, l, leaf_d, threads_per);
            }
        }        
    }
}

//Assumes 2^{d-1} - 1 < n < 2^d - 1
template<typename TYPE>
void permutevEB_nonperfect(TYPE *A, uint64_t n, uint32_t d) {
    if (d == 1) return;
    else {
        uint32_t root_d = (d - 2)/2 + 1;        //floor((d - 2)/2) + 1
        uint32_t leaf_d = d - root_d;           //ceil((d - 2)/2.) + 1

        uint64_t r = pow(2, root_d) - 1;        //number of elements in root subtree
        uint64_t l = pow(2, leaf_d) - 1;        //number of elements in leaf subtrees

        uint64_t num_full = (n - r) / l;        //number of full leaf subtrees
        uint64_t inc_n = n - r - num_full*l;    //number of nodes in the incomplete leaf subtree

        //Gather root elements to the front of the array
        uint64_t temp_n = num_full*(l+1);
        unshuffle_dk<TYPE>(A, l+1, temp_n);
        shift_right<TYPE>(A, temp_n, num_full);
        shuffle_dk<TYPE>(&A[num_full], l, temp_n - num_full);

        if (num_full < r) {
            shift_right<TYPE>(&A[num_full], n - num_full, r - num_full);
        }

        //Recurse
        uint64_t size;
        if (root_d == leaf_d) {
            size = (num_full + 1)*r;
            for (uint64_t i = 0; i < size; i += r) {        //Recurse on root and full leaf subtrees
                permutevEB<TYPE>(&A[i], r, root_d);
            }
        }
        else {
            permutevEB<TYPE>(A, r, root_d);     //Recurse on root subtree

            size = r + num_full*l;
            for (uint64_t i = r; i < size; i += l) {        //Recurse on full leaf subtrees
                permutevEB<TYPE>(&A[i], l, leaf_d);
            }
        }

        if (inc_n > 0) {
            uint32_t inc_d = log2(inc_n) + 1;

            //Recurse on incomplete leaf subtree
            if (inc_n != pow(2, inc_d) - 1) {       //non-perfect incomplete tree
                permutevEB_nonperfect<TYPE>(&A[size], inc_n, inc_d);
            }
            else {      //perfect incomplete tree
                permutevEB<TYPE>(&A[size], inc_n, inc_d);
            } 
        }
    }
}

//Assumes 2^{d-1} - 1 < n < 2^d - 1
template<typename TYPE>
void permutevEB_nonperfect_parallel(TYPE *A, uint64_t n, uint32_t d, uint32_t p) {
    if (d == 1) return;
    else {
        uint32_t root_d = (d - 2)/2 + 1;        //floor((d - 2)/2) + 1
        uint32_t leaf_d = d - root_d;           //ceil((d - 2)/2.) + 1

        uint64_t r = pow(2, root_d) - 1;        //number of elements in the root subtree
        uint64_t l = pow(2, leaf_d) - 1;        //number of elements in the full leaf subtrees

        uint64_t num_full = (n - r) / l;        //number of full leaf subtrees
        uint64_t inc_n = n - r - num_full*l;    //number of nodes in the incomplete leaf subtree

        //Gather root elements to the front of the array
        uint64_t temp_n = num_full*(l+1);
        unshuffle_dk_parallel<TYPE>(A, l+1, temp_n, p);
        shift_right_parallel<TYPE>(A, temp_n, num_full, p);
        shuffle_dk_parallel<TYPE>(&A[num_full], l, temp_n - num_full, p);

        if (num_full < r) {
            shift_right_parallel<TYPE>(&A[num_full], n - num_full, r - num_full, p);
        }
        
        //Recurse
        //Parallel solution #1
        uint64_t size;
        if (root_d == leaf_d) {
            size = (num_full + 1)*r;

            if (p <= num_full + 1) {
                #pragma omp parallel for shared(A, n, d, p, root_d, leaf_d, r, l, num_full, inc_n, size) schedule(guided) num_threads(p)
                for (uint64_t i = 0; i < size; i += r) {        //Recurse on root and full leaf subtrees
                    permutevEB<TYPE>(&A[i], r, root_d);
                }
            }
            else {
                uint32_t threads_per = ceil(p/(double)(num_full + 1));

                #pragma omp parallel for shared(A, n, d, p, root_d, leaf_d, r, l, num_full, inc_n, size, threads_per) schedule(guided) num_threads(num_full+1)
                for (uint64_t i = 0; i < size; i += r) {        //Recurse on root and full leaf subtrees
                    permutevEB_parallel<TYPE>(&A[i], r, root_d, threads_per);
                }
            }
        }
        else {
            permutevEB_parallel<TYPE>(A, r, root_d, p);         //Recurse on root subtree

            size = r + num_full*l;

            if (p <= num_full) {
                #pragma omp parallel for shared(A, n, d, p, root_d, leaf_d, r, l, num_full, inc_n, size) schedule(guided) num_threads(p)
                for (uint64_t i = r; i < size; i += l) {        //Recurse on full leaf subtrees
                    permutevEB<TYPE>(&A[i], l, leaf_d);
                }
            }
            else {
                uint32_t threads_per = ceil(p/(double)num_full);

                #pragma omp parallel for shared(A, n, d, p, root_d, leaf_d, r, l, num_full, inc_n, size, threads_per) schedule(guided) num_threads(num_full)
                for (uint64_t i = r; i < size; i += l) {        //Recurse on full leaf subtrees
                    permutevEB_parallel<TYPE>(&A[i], l, leaf_d, threads_per);
                }
            }
        }

        if (inc_n > 0) {
            uint32_t inc_d = log2(inc_n) + 1;

            //Recurse on incomplete leaf subtree
            if (inc_n != pow(2, inc_d) - 1) {       //non-perfect incomplete tree
                permutevEB_nonperfect_parallel<TYPE>(&A[size], inc_n, inc_d, p);
            }
            else {      //perfect incomplete tree
                permutevEB_parallel<TYPE>(&A[size], inc_n, inc_d, p);
            } 
        }        

        //Parallel Solution #2: slightly slower than #1
        /*if (root_d == leaf_d) {
            uint64_t size = (num_full + 1)*r;

            if (inc_n > 0) {
                uint32_t inc_d = log2(inc_n) + 1;

                if (p <= num_full + 2) {
                    //printf("case 1\n");
                    #pragma omp parallel for shared(A, n, d, p, root_d, leaf_d, r, l, num_full, inc_n, inc_d, size) schedule(guided) num_threads(p)
                    for (uint64_t i = 0; i < n; i += r) {
                        if (i < size) permutevEB<TYPE>(&A[i], r, root_d);       //Recurse on root and full leaf subtrees
                        else {
                            //Recurse on incomplete leaf subtree
                            if (inc_n != pow(2, inc_d) - 1) {       //non-perfect incomplete tree
                                permutevEB_nonperfect<TYPE>(&A[size], inc_n, inc_d);
                            }
                            else {      //perfect incomplete tree
                                permutevEB<TYPE>(&A[size], inc_n, inc_d);
                            } 
                        }
                    }
                }
                else {
                    //printf("case 2\n");
                    uint32_t threads_per = ceil(p/(double)(num_full + 2));

                    #pragma omp parallel for shared(A, n, d, p, root_d, leaf_d, r, l, num_full, inc_n, inc_d, size) schedule(guided) num_threads(num_full+2)
                    for (uint64_t i = 0; i < n; i += r) {
                        if (i < size) permutevEB_parallel<TYPE>(&A[i], r, root_d, threads_per);       //Recurse on root and full leaf subtrees
                        else {
                            //Recurse on incomplete leaf subtree
                            if (inc_n != pow(2, inc_d) - 1) {       //non-perfect incomplete tree
                                permutevEB_nonperfect_parallel<TYPE>(&A[size], inc_n, inc_d, threads_per);
                            }
                            else {      //perfect incomplete tree
                                permutevEB_parallel<TYPE>(&A[size], inc_n, inc_d, threads_per);
                            } 
                        }
                    }
                }
            }
            else {
                uint64_t size = (num_full + 1)*r;

                if (p <= num_full + 1) {
                    //printf("case 3\n");
                    #pragma omp parallel for shared(A, n, d, p, root_d, leaf_d, r, l, num_full, inc_n, size) schedule(guided) num_threads(p)
                    for (uint64_t i = 0; i < size; i += r) {        //Recurse on root and full leaf subtrees
                        permutevEB<TYPE>(&A[i], r, root_d);
                    }
                }
                else {
                    //printf("case 4\n");
                    uint32_t threads_per = ceil(p/(double)(num_full + 1));

                    #pragma omp parallel for shared(A, n, d, p, root_d, leaf_d, r, l, num_full, inc_n, size, threads_per) schedule(guided) num_threads(num_full+1)
                    for (uint64_t i = 0; i < size; i += r) {        //Recurse on root and full leaf subtrees
                        permutevEB_parallel<TYPE>(&A[i], r, root_d, threads_per);
                    }
                }
            }
        }
        else {
            if (inc_n > 0) {
                uint32_t inc_d = log2(inc_n) + 1;

                permutevEB_parallel<TYPE>(A, r, root_d, p);         //Recurse on root subtree

                uint64_t size = r + num_full*l;

                if (p <= num_full + 1) {
                    //printf("case 5: inc_n = %lu\n", inc_n);
                    #pragma omp parallel for shared(A, n, d, p, root_d, leaf_d, r, l, num_full, inc_n, inc_d, size) schedule(guided) num_threads(p)
                    for (uint64_t i = r; i < n; i += l) {
                        if (i < size) permutevEB<TYPE>(&A[i], l, leaf_d);       //Recurse on full leaf subtrees
                        else {
                            //Recurse on incomplete leaf subtree
                            if (inc_n != pow(2, inc_d) - 1) {       //non-perfect incomplete tree
                                permutevEB_nonperfect<TYPE>(&A[size], inc_n, inc_d);
                            }
                            else {      //perfect incomplete tree
                                permutevEB<TYPE>(&A[size], inc_n, inc_d);
                            }
                        }
                    }
                }
                else {
                    //printf("case 6\n");
                    uint32_t threads_per = ceil(p/(double)(num_full + 1));

                    #pragma omp parallel for shared(A, n, d, p, root_d, leaf_d, r, l, num_full, inc_n, inc_d, size, threads_per) schedule(guided) num_threads(num_full)
                    for (uint64_t i = r; i < n; i += l) {
                        if (i < size) permutevEB_parallel<TYPE>(&A[i], l, leaf_d, threads_per);       //Recurse on full leaf subtrees
                        else {
                            //Recurse on incomplete leaf subtree
                            if (inc_n != pow(2, inc_d) - 1) {       //non-perfect incomplete tree
                                permutevEB_nonperfect_parallel<TYPE>(&A[size], inc_n, inc_d, p);
                            }
                            else {      //perfect incomplete tree
                                permutevEB_parallel<TYPE>(&A[size], inc_n, inc_d, p);
                            }
                        }
                    }
                }
            }
            else {
                permutevEB_parallel<TYPE>(A, r, root_d, p);         //Recurse on root subtree

                uint64_t size = r + num_full*l;

                if (p <= num_full) {
                    //printf("case 7\n");
                    #pragma omp parallel for shared(A, n, d, p, root_d, leaf_d, r, l, num_full, inc_n, size) schedule(guided) num_threads(p)
                    for (uint64_t i = r; i < size; i += l) {        //Recurse on full leaf subtrees
                        permutevEB<TYPE>(&A[i], l, leaf_d);
                    }
                }
                else {
                    //printf("case 8\n");
                    uint32_t threads_per = ceil(p/(double)num_full);

                    #pragma omp parallel for shared(A, n, d, p, root_d, leaf_d, r, l, num_full, inc_n, size, threads_per) schedule(guided) num_threads(num_full)
                    for (uint64_t i = r; i < size; i += l) {        //Recurse on full leaf subtrees
                        permutevEB_parallel<TYPE>(&A[i], l, leaf_d, threads_per);
                    }
                }
            }
        }*/
    }
}

template<typename TYPE>
double timePermutevEB(TYPE *A, uint64_t n, uint32_t p) {
	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC, &start);

    uint32_t h = log2(n);
    if (n != pow(2, h+1) - 1) {     //non-full tree
        if (p == 1) permutevEB_nonperfect<TYPE>(A, n, h+1);
        else permutevEB_nonperfect_parallel<TYPE>(A, n, h+1, p);
    }
    else {    //full tree
        if (p == 1) permutevEB(A, n, h+1);
        else permutevEB_parallel<TYPE>(A, n, h+1, p);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double ms = ((end.tv_sec*1000000000. + end.tv_nsec) - (start.tv_sec*1000000000. + start.tv_nsec)) / 1000000.;		//millisecond
    return ms;
}
#endif