#ifndef MVEB_CYCLES_H
#define MVEB_CYCLES_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#include "common.h"
#include "cycles.h"

//Permutes sorted array into the van Emde Boas tree layout via equidistant gather
//Both the top/root and bottom/leaf subtrees have a height of (h - 1)/2
//I.e., the tree is perfectly balanced (n is a power of power of 2 minus 1 <==> d is a power of 2)
template<typename TYPE>
void permutemvEB_balanced(TYPE *A, uint64_t n, uint32_t d) {
    if (n == 1) return;

    uint64_t m = (uint64_t)sqrt(n);     //floor{sqrt(n)} = 2^{d/2} - 1

    equidistant_gather_io<TYPE>(A, m, m);
    
    //Recurse on each subtree
    for (uint64_t i = 0; i < n; i += m) {
        permutemvEB_balanced<TYPE>(&A[i], m, d/2);
    }
}

//Permutes sorted array into the van Emde Boas tree layout via equidistant gather
//Both the top/root and bottom/leaf subtrees have a height of (h - 1)/2
//I.e., the tree is perfectly balanced (n is a power of power of 2 minus 1 <==> d is a power of 2)
template<typename TYPE>
void permutemvEB_balanced_parallel(TYPE *A, uint64_t n, uint32_t d, uint32_t p) {
    if (n == 1) return;

    uint64_t m = (uint64_t)sqrt(n);     //floor{sqrt(n)} = 2^{d/2} - 1

    equidistant_gather_io_parallel<TYPE>(A, m, m, p);
    
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

//Permutes sorted array into the van Emde Boas tree layout via equidistant gather
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

    if (r <= l) {
        equidistant_gather_io<TYPE>(A, r, l);
    }
    else {      //in some cases, r = 2*l + 1 as in mvEB; e.g., n = 7, r = 3, l = 1
        equidistant_gather_io<TYPE>(A, l, l);
        equidistant_gather_io<TYPE>(&A[n/2 + 1], l, l);
        shift_right<TYPE>(&A[l], (l+1)*(l+1), l+1);
    }
    
    permutemvEB<TYPE>(A, r, root_d);               //recurse on root subtree

    uint32_t numLeafTrees = (n - r)/l;
    for (int i = 0; i < numLeafTrees; i++) {
        permutemvEB_balanced<TYPE>(&A[r + i*l], l, leaf_d);        //recurse on i-th leaf subtree
    }
}

//Permutes sorted array into the van Emde Boas tree layout via equidistant gather using p processors
//The leaf subtrees have a height of floor{(h - 1)/2} rounded up to the nearest power of 2 minus 1 (depth is rounded to a power of 2)
//I.e., the number of nodes in each leaf subtree is a power of power of 2 minus 1
template<typename TYPE>
void permutemvEB_parallel(TYPE *A, uint64_t n, uint32_t d, uint32_t p) {
    if (n == 1) return;
    
    uint32_t leaf_d = (d - 2)/2 + 1;

    float log_leaf = log2((float)leaf_d);

    if (log_leaf - ((int)log_leaf) != 0) {      //Not a perfectly balanced mvEB
        leaf_d = pow(2, ceil(log_leaf));        //Round up to nearest power of 2
    }

    uint32_t root_d = d - leaf_d;

    uint64_t r = pow(2, root_d) - 1;
    uint64_t l = pow(2, leaf_d) - 1;

    if (r <= l) {
        equidistant_gather_io_parallel<TYPE>(A, r, l, p);
    }
    else {      //in some cases, r = 2*l + 1 as in mvEB; e.g., n = 7, r = 3, l = 1
        equidistant_gather_io_parallel<TYPE>(A, l, l, p);
        equidistant_gather_io_parallel<TYPE>(&A[n/2 + 1], l, l, p);
        shift_right_parallel<TYPE>(&A[l], (l+1)*(l+1), l+1, p);
    }

    //Recurse on root subtree
    permutemvEB_parallel<TYPE>(A, r, root_d, p);

    //Recurse on each leaf subtree in parallel
    uint64_t numLeafTrees = (n - r)/l;
    if (p <= numLeafTrees) {
        #pragma omp parallel for shared(A, n, numLeafTrees, r, l, leaf_d) schedule(guided) num_threads(p)
        for (uint64_t i = 0; i < numLeafTrees; ++i) {     
            permutemvEB_balanced<TYPE>(&A[r + i*l], l, leaf_d);
        }
    }
    else {
        uint32_t threads_per = ceil(p/(double)numLeafTrees);

        #pragma omp parallel for shared(A, n, numLeafTrees, r, l, leaf_d, threads_per) num_threads(numLeafTrees)
        for (uint64_t i = 0; i < numLeafTrees; ++i) {     
            permutemvEB_balanced_parallel<TYPE>(&A[r + i*l], l, leaf_d, threads_per);
        }
    }
}

template<typename TYPE>
double timePermutemvEB(TYPE *A, uint64_t n, uint32_t p) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    uint32_t d = log2(n) + 1;
    if (n != pow(2, d) - 1) {     //non-full tree
        printf("Non-perfect mvEB tree ==> NOT YET IMPLEMENTED!\n");
        return 0.;
    }
    else {    //full tree
        float log_d = log2((float)d);
        if (p == 1) {
            if (log_d - ((int)log_d) == 0) permutemvEB_balanced<TYPE>(A, n, d);     //d is a power of 2 <==> n is a power of power of 2 minus 1
            else permutemvEB(A, n, d);
        }
        else {
            if (log_d - ((int)log_d) == 0) permutemvEB_balanced_parallel<TYPE>(A, n, d, p);     //d is a power of 2 <==> n is a power of power of 2 minus 1
            else permutemvEB_parallel(A, n, d, p);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double ms = ((end.tv_sec*1000000000. + end.tv_nsec) - (start.tv_sec*1000000000. + start.tv_nsec)) / 1000000.;       //millisecond
    return ms;
}
#endif