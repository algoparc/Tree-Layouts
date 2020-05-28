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

#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#include "params.h"

void printA(uint64_t *A, uint64_t n);
void printTimings(uint64_t n, double time[ITERS], int p);
void printQueryTimings(uint64_t n, uint64_t q, double time[ITERS], int p);

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