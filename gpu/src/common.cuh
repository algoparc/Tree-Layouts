#ifndef COMMON_CUH
#define COMMON_CUH

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include <time.h>

#include "params.cuh"

void printA(uint64_t *A, uint64_t n);
void printTimings(uint64_t n, double time[ITERS]);
void printQueryTimings(uint64_t n, uint64_t q, double time[ITERS]);

//Populates the given array of size with 1, 2, ..., n casted to the given generic type
template<typename TYPE>
void initSortedList(TYPE *data, uint64_t n) {
    #pragma omp parallel for
    for (uint64_t i = 0; i < n; i++) {
        data[i] = (TYPE)(i+1);
    }
}

//Creates random queries for the given data array of n elements
//Returns a pointer to an array containing the generated random queries of size numQueries
template<typename TYPE>
TYPE *createRandomQueries(TYPE *data, uint64_t n, uint64_t numQueries) {
    TYPE *queries = (TYPE *)malloc(numQueries * sizeof(TYPE));

    #ifdef DEBUG
    srand(0);
    #else
    srand(time(NULL));
    #endif
  
    #pragma omp parallel for
    for (uint64_t i = 0; i < numQueries; i++) {
        queries[i] = (TYPE)data[rand() % n];
    }

    return queries;
}
#endif