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
//The top subtree has a height of ceil{(h - 1)/2} and leaf subtrees have height of floor{(h - 1)/2}
template<typename TYPE>
void permutevEB(TYPE *A, uint64_t n, uint32_t d) {
	if (n == 1) return;

    uint32_t leaf_d = (d - 2)/2 + 1;		//floor((d - 2)/2) + 1
    uint32_t root_d = d - leaf_d;			//ceil((d - 2)/2.) + 1

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

    uint32_t leaf_d = (d - 2)/2 + 1;        //floor((d - 2)/2) + 1
    uint32_t root_d = d - leaf_d;           //ceil((d - 2)/2.) + 1

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
    
    if (d % 2 == 0) {
        uint64_t m = (uint64_t)sqrt(n);     //floor{sqrt(n)} = 2^{d/2} - 1
        
        if (n/m >= p) {     //if number of recursive calls is larger than p, have each processor sequentially permute in parallel
            #pragma omp parallel for shared(A, n, d, p, m) schedule(guided) num_threads(p)
            for (uint64_t i = 0; i < n; i += m) {        //Recurse on each subtree
                permutevEB<TYPE>(&A[i], m, d/2);
            }
        }
        else {      //else number of processors available is larger than number of recursive calls
            uint32_t threads_per = ceil(p/(double)(n/m));

            #pragma omp parallel for shared(A, n, d, p, m, threads_per) schedule(guided) num_threads(n/m)
            for (uint64_t i = 0; i < n; i += m) {        //Recurse on each subtree
                permutevEB_parallel<TYPE>(&A[i], m, d/2, threads_per);
            }
        }
    }
    else {
        //Parallel solution #1
        //Recurse on root subtree
        permutevEB_parallel<TYPE>(A, 2*l + 1, d/2 + 1, p);

        //Recurse on leaf subtress in parallel
        uint64_t numLeafTrees = (n - (2*l + 1)) / l;
        if (p <= numLeafTrees) {
            #pragma omp parallel for shared(A, n, d, p, l, numLeafTrees) schedule(guided) num_threads(p)
            for (uint64_t i = 2*l + 1; i < n; i += l) {
                permutevEB<TYPE>(&A[i], l, d/2);
            }
        }
        else {
            uint32_t threads_per = ceil(p/(double)numLeafTrees);
                    
            #pragma omp parallel for shared(A, n, d, p, l, numLeafTrees, threads_per) num_threads(numLeafTrees)
            for (uint64_t i = 2*l + 1; i < n; i += l) {
                permutevEB_parallel<TYPE>(&A[i], l, d/2, threads_per);
            }
        }

        /*//Parallel solution #2
        #pragma omp parallel sections num_threads(2)
        {
            #pragma omp section
            {
                uint32_t root_p = ceil((p/(double)n)*(2*l + 1));        //scale p by number of elements in the root subtree

                //Recurse on root subtree
                if (root_p == 1) permutevEB<TYPE>(A, 2*l + 1, d/2 + 1);
                else permutevEB_parallel<TYPE>(A, 2*l + 1, d/2 + 1, root_p);       
            }
            #pragma omp section
            {
                uint64_t numLeafTrees = (n - (2*l + 1)) / l;
                uint32_t leaf_p = ceil((p/(double)n)*(n - 2*l - 1));      //scale p by the number of of elements in all leaf subtrees

                //Recurse on each leaf subtree
                if (leaf_p == 1) {
                    for (uint64_t i = 2*l + 1; i < n; i += l) {     
                        permutevEB<TYPE>(&A[i], l, d/2);
                    }
                }               
                else if (numLeafTrees >= leaf_p) {
                    #pragma omp parallel for shared(A, n, d, leaf_p, l, numLeafTrees) schedule(guided) num_threads(leaf_p)
                    for (uint64_t i = 2*l + 1; i < n; i += l) {
                        permutevEB<TYPE>(&A[i], l, d/2);
                    }
                }
                else {
                    uint32_t threads_per = ceil(leaf_p/(double)numLeafTrees);
                    
                    #pragma omp parallel for shared(A, n, d, leaf_p, l, numLeafTrees, threads_per) num_threads(numLeafTrees)
                    for (uint64_t i = 2*l + 1; i < n; i += l) {
                        permutevEB_parallel<TYPE>(&A[i], l, d/2, threads_per);
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
        printf("non-perfect vEB tree ==> NOT YET IMPLEMENTED!\n");
        return 0.;
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