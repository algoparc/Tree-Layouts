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

#ifndef VEB_CYCLES_H
#define VEB_CYCLES_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#include "common.h"
#include "cycles.h"

//Permutes sorted array into the van Emde Boas tree layout via Level-order B-tree involutions
//B = # of leaf elements per leaf subtree, i.e., paramter l (in the below code)
//The top subtree has a height of ceil{(h - 1)/2} and leaf subtrees have height of floor{(h - 1)/2}
template<typename TYPE>
void permutevEB(TYPE *A, uint64_t n, uint32_t d) {
	if (d == 1) return;
    else if (d % 2 == 0) {       //balanced, |T_{root}| = |T_{leaf}|
        uint64_t m = (uint64_t)sqrt(n);     //floor{sqrt(n)} = 2^{d/2} - 1

        equidistant_gather_io<TYPE>(A, m, m);

        for (uint64_t i = 0; i < n; i+= m) {        //Recurse on each subtree
            permutevEB<TYPE>(&A[i], m, d/2);
        }
    }
    else {      //unbalanced, |T_{leaf}| = 2*|T_{root}| + 1
        uint64_t r = pow(2, d/2) - 1;
        uint64_t l = 2*r + 1;

        equidistant_gather_io<TYPE>(A, r, l);

        permutevEB<TYPE>(A, r, d/2);                //Recurse on root subtree
        for (uint64_t i = r; i < n; i += l) {       //Recurse on each leaf subtree
            permutevEB<TYPE>(&A[i], l, d/2 + 1);
        }
    }
}

//Permutes sorted array into the van Emde Boas tree layout via Level-order B-tree involutions
//B = # of leaf elements per leaf subtree, i.e., paramter l (in the below code)
//The top subtree has a height of ceil{(h - 1)/2} and leaf subtrees have height of floor{(h - 1)/2}
template<typename TYPE>
void permutevEB_parallel(TYPE *A, uint64_t n, uint32_t d, uint32_t p) {
    if (n == 1) return;
    else if (d % 2 == 0) {       //balanced, |T_{root}| = |T_{leaf}|
        uint64_t m = (uint64_t)sqrt(n);     //floor{sqrt(n)} = 2^{d/2} - 1

        equidistant_gather_io_parallel<TYPE>(A, m, m, p);

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
    else {      //unbalanced, |T_{leaf}| = 2*|T_{root}| + 1
        uint64_t r = pow(2, d/2) - 1;
        uint64_t l = 2*r + 1;

        equidistant_gather_io_parallel<TYPE>(A, r, l, p);

        permutevEB_parallel<TYPE>(A, r, d/2, p);

        uint64_t numLeafTrees = (n-r)/l;
        if (p <= numLeafTrees) {
            #pragma omp parallel for shared(A, n, d, p, r, l, numLeafTrees) schedule(guided) num_threads(p)
            for (uint64_t i = r; i < n; i += l) {       //Recurse on each leaf subtree
                permutevEB<TYPE>(&A[i], l, d/2 + 1);
            }
        }
        else {
            uint32_t threads_per = ceil(p/(double)numLeafTrees);

            #pragma omp parallel for shared(A, n, d, p, r, l, numLeafTrees, threads_per) schedule(guided) num_threads(numLeafTrees)
            for (uint64_t i = r; i < n; i += l) {       //Recurse on each leaf subtree
                permutevEB_parallel<TYPE>(&A[i], l, d/2 + 1, threads_per);
            }
        }
    }
}

//Assumes 2^{d-1} - 1 < n < 2^d - 1
template<typename TYPE>
void permutevEB_nonperfect(TYPE *A, uint64_t n, uint32_t d) {
    //#ifdef DEBUG
    //printf("d = %u; n = %lu ==> ", d, n);
    //#endif

    if (d == 1) return;
    else {
        uint32_t root_d = (d - 2)/2 + 1;        //floor((d - 2)/2) + 1
        uint32_t leaf_d = d - root_d;           //ceil((d - 2)/2.) + 1

        uint64_t r = pow(2, root_d) - 1;        //number of elements in the root subtree
        uint64_t l = pow(2, leaf_d) - 1;        //number of elements in the full leaf subtrees

        uint64_t num_full = (n - r) / l;        //number of full leaf subtrees
        uint64_t inc_n = n - r - num_full*l;    //number of nodes in the incomplete leaf subtree

        //#ifdef DEBUG
        //printf("root_d = %u; leaf_d = %u; r = %lu; l = %lu\n", root_d, leaf_d, r, l);
        //printf("num_full = %lu; inc_n = %lu\n", num_full, inc_n);
        //#endif

        //Gather root elements to the front of the array
        equidistant_gather_io<TYPE>(A, num_full, l);
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

            //#ifdef DEBUG
            //printf("inc_d = %u\n\n", inc_d);
            //#endif

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
        equidistant_gather_io_parallel<TYPE>(A, num_full, l, p);
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
        if (p == 1) permutevEB<TYPE>(A, n, h+1);
        else permutevEB_parallel<TYPE>(A, n, h+1, p);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double ms = ((end.tv_sec*1000000000. + end.tv_nsec) - (start.tv_sec*1000000000. + start.tv_nsec)) / 1000000.;		//millisecond   
    return ms;
}
#endif