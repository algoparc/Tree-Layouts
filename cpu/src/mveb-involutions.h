#ifndef MVEB_INVOLUTIONS_H
#define MVEB_INVOLUTIONS_H

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
//Both the top/root and bottom/leaf subtrees have a height of (h - 1)/2
//I.e., the tree is perfectly balanced (n is a power of power of 2 minus 1 <==> d is a power of 2)
template<typename TYPE>
void permutemvEB_balanced(TYPE *A, uint64_t n, uint32_t d) {
    if (n == 1) return;

    uint64_t m = (uint64_t)sqrt(n);     //floor{sqrt(n)} = 2^{d/2} - 1

    uint32_t h = log10(n)/log10(m+1);
    if (n != pow(m+1, h+1) - 1) {
        //printf("non-perfect B-tree of height %ld\n", h);
        unshuffle_dk<TYPE>(A - 1, m+1, n + 1);      
        shuffle_dk<TYPE>(&A[m], m, n - m);
    }
    else {
        //printf("perfect B-tree of height %ld\n", h);
        unshuffle<TYPE>(A, m+1, n, h+1);
        shuffle_dk<TYPE>(&A[n/(m+1)], m, pow(m+1, h) * m);
    }
    
    //Recurse on each subtree
    for (uint64_t i = 0; i < n; i += m) {
        permutemvEB_balanced<TYPE>(&A[i], m, d/2);
    }
}

//Permutes sorted array into the van Emde Boas tree layout via Level-order B-tree involutions using p processors
//B = # of leaf elements per leaf subtree, i.e., paramter l (in the below code)
//Both the top/root and bottom/leaf subtrees have a height of (h - 1)/2
//I.e., the tree is perfectly balanced (n is a power of power of 2 minus 1 <==> d is a power of 2)
template<typename TYPE>
void permutemvEB_balanced_parallel(TYPE *A, uint64_t n, uint32_t d, uint32_t p) {
    if (n == 1) return;

    uint64_t m = (uint64_t)sqrt(n);     //floor{sqrt(n)} = 2^{d/2} - 1

    uint32_t h = log10(n)/log10(m+1);
    if (n != pow(m+1, h+1) - 1) {
        //printf("non-perfect B-tree of height %ld\n", h);
        unshuffle_dk_parallel<TYPE>(A - 1, m+1, n + 1, p);
        shuffle_dk_parallel<TYPE>(&A[m], m, n - m, p);
    }
    else {
        //printf("perfect B-tree of height %ld\n", h);
        unshuffle_parallel<TYPE>(A, m+1, n, h+1, p);
        shuffle_dk_parallel<TYPE>(&A[n/(m+1)], m, pow(m+1, h) * m, p);
    }

    if (n/m >= p) {
        //Recurse on each subtree
        #pragma omp parallel for shared(A, n, d, p, m) schedule(guided) num_threads(p)
        for (uint64_t i = 0; i < n; i += m) {
            permutemvEB_balanced<TYPE>(&A[i], m, d/2);
        }
    }
    else {
        uint32_t threads_per = ceil(p/(double)(n/m));

        //Recurse on each subtree
        #pragma omp parallel for shared(A, n, d, p, m, threads_per) num_threads(n/m)
        for (uint64_t i = 0; i < n; i += m) {
            permutemvEB_balanced_parallel<TYPE>(&A[i], m, d/2, threads_per);
        }
    }
}

//Permutes sorted array into the van Emde Boas tree layout via Level-order B-tree involutions
//b = # of leaf elements per leaf subtree, i.e., parameter l (in the below code)
//The leaf subtrees have a height of floor{(h - 1)/2} rounded up to the nearest power of 2 minus 1 (depth is rounded to a power of 2)
//I.e., the number of nodes in each leaf subtree is a power of power of 2 minus 1
template<typename TYPE>
void permutemvEB(TYPE *A, uint64_t n, uint32_t d) {
    if (n == 1) return;  
    
    uint32_t leaf_d = (d - 2)/2 + 1;
    float log_leaf = log2((float)leaf_d);
    if (log_leaf - ((int)log_leaf) != 0) {      //Not a perfectly balanced mvEB
        leaf_d = pow(2, ceil(log_leaf));
    }

    uint32_t root_d = d - leaf_d;

    uint64_t r = pow(2, root_d) - 1;
    uint64_t l = pow(2, leaf_d) - 1;

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
    
    permutemvEB<TYPE>(A, r, root_d);               //recurse on root subtree

    uint32_t numLeafTrees = (n - r)/l;
    for (int i = 0; i < numLeafTrees; i++) {
        permutemvEB_balanced<TYPE>(&A[r + i*l], l, leaf_d);        //recurse on i-th leaf subtree
    }
}

//Permutes sorted array into the van Emde Boas tree layout via Level-order B-tree involutions using p processors
//b = # of leaf elements per leaf subtree, i.e., parameter l (in the below code)
//The leaf subtrees have a height of floor{(h - 1)/2} rounded up to the nearest power of 2 minus 1 (depth is rounded to a power of 2)
//I.e., the number of nodes in each leaf subtree is a power of power of 2 minus 1
template<typename TYPE>
void permutemvEB_parallel(TYPE *A, uint64_t n, uint32_t d, uint32_t p) {
    if (n == 1) return;  
    
    uint32_t leaf_d = (d - 2)/2 + 1;
    float log_leaf = log2((float)leaf_d);
    if (log_leaf - ((int)log_leaf) != 0) {      //Not a perfectly balanced mvEB
        leaf_d = pow(2, ceil(log_leaf));
    }

    uint32_t root_d = d - leaf_d;

    uint64_t r = pow(2, root_d) - 1;
    uint64_t l = pow(2, leaf_d) - 1;

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
    
    //Parallel Solution #1
    //Recurse on root subtree
    permutemvEB_parallel<TYPE>(A, r, root_d, p);

    //Recurse on each leaf subtree in parallel
    uint32_t numLeafTrees = (n - r)/l;
    if (p <= numLeafTrees) {
        #pragma omp parallel for shared(A, n, d, p, r, l, leaf_d, numLeafTrees) schedule(guided) num_threads(p)
        for (uint64_t i = 0; i < numLeafTrees; ++i) {
            permutemvEB_balanced<TYPE>(&A[r + i*l], l, leaf_d);
        }
    }
    else {
        uint32_t threads_per = ceil(p/(double)numLeafTrees);

        #pragma omp parallel for shared(A, n, d, p, r, l, leaf_d, numLeafTrees, threads_per) num_threads(numLeafTrees)
        for (uint64_t i = 0; i < numLeafTrees; ++i) {
            permutemvEB_balanced_parallel<TYPE>(&A[r + i*l], l, leaf_d, threads_per);
        }
    }

    /*//Parallel Solution #2
    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        {
            uint32_t root_p = ceil((p/(double)n)*r);

            //Recurse on root subtree
            if (root_p == 1) permutemvEB<TYPE>(A, r, root_d);
            else permutemvEB_parallel<TYPE>(A, r, root_d, root_p);
        }
        #pragma omp section
        {
            uint32_t leaf_p = ceil((p/(double)n)*(n - r));
            uint32_t numLeafTrees = (n - r)/l;

            //Recurse on each leaf subtree
            if (leaf_p == 1) {
                for (uint64_t i = 0; i < numLeafTrees; ++i) {
                    permutemvEB_balanced<TYPE>(&A[r + i*l], l, leaf_d);
                }
            }
            else if (numLeafTrees >= leaf_p) {
                #pragma omp parallel for shared(A, n, d, leaf_p, numLeafTrees, r, l, leaf_d) schedule(guided) num_threads(leaf_p)
                for (uint64_t i = 0; i < numLeafTrees; ++i) {
                    permutemvEB_balanced<TYPE>(&A[r + i*l], l, leaf_d);
                }
            }
            else {
                uint32_t threads_per = ceil(leaf_p/(double)numLeafTrees);

                #pragma omp parallel for shared(A, n, d, leaf_p, numLeafTrees, r, l, leaf_d) num_threads(numLeafTrees)
                for (uint64_t i = 0; i < numLeafTrees; ++i) {
                    permutemvEB_balanced_parallel<TYPE>(&A[r + i*l], l, leaf_d, threads_per);
                }
            }
        }
    }*/
}

template<typename TYPE>
double timePermutemvEB(TYPE *A, uint64_t n, uint32_t p) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    uint32_t d = log2(n) + 1;
    if (n != pow(2, d) - 1) {     //non-full tree
        printf("Non-perfect mvEB tree ==> NOT YET IMPLEMENTED\n");
        return 0.;
    }
    else {    //full tree
        float log_d = log2((float)d);
        if (p == 1) {
            if (log_d - ((int)log_d) == 0) permutemvEB_balanced<TYPE>(A, n, d);     //d is a power of 2 <==> n is a power of power of 2 minus 1
            else permutemvEB<TYPE>(A, n, d);
        }
        else {
            if (log_d - ((int)log_d) == 0) permutemvEB_balanced_parallel<TYPE>(A, n, d, p);     //d is a power of 2 <==> n is a power of power of 2 minus 1
            else permutemvEB_parallel<TYPE>(A, n, d, p);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double ms = ((end.tv_sec*1000000000. + end.tv_nsec) - (start.tv_sec*1000000000. + start.tv_nsec)) / 1000000.;       //millisecond
    return ms;
}
#endif