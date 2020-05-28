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

#ifndef BINARY_SEARCH_H
#define BINARY_SEARCH_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#include "common.h"

//Performs binary search on the sorted array A of size n
template<typename TYPE>
uint64_t binarySearch(TYPE *A, uint64_t n, TYPE query) {
	uint64_t i = n/2;

	for (uint64_t step = n/4 + 1; step >= 1; step /= 2) {
		if (query == A[i]) {
			return i;
		}
		else if (query > A[i]) {
			i += step;
		}
		else {
			i -= step;
		}
	}
	return i;
}

//Performs all of the queries given in the array queries
//index in A of the queried items are saved in the answers array
template<typename TYPE>
void searchAll(TYPE *A, uint64_t n, TYPE *queries, uint64_t *answers, uint64_t numQueries, uint32_t p) {
    #pragma omp parallel for shared(A, n, queries, answers, numQueries, p) schedule(guided) num_threads(p)
    for (uint64_t i = 0; i < numQueries; ++i) {
        answers[i] = binarySearch<TYPE>(A, n, queries[i]);
    }
}

//Generates numQueries random queries and returns the milliseconds needed to perform the queries
template<typename TYPE>
double timeQuery(TYPE *A, uint64_t n, uint64_t numQueries, uint32_t p) {
    struct timespec start, end;

    TYPE *queries = createRandomQueries<TYPE>(A, n, numQueries);              //array to store random queries to perform
    uint64_t *answers = (uint64_t *)malloc(numQueries * sizeof(uint64_t));    //array to store the answers (i.e., index of the queried item)

    clock_gettime(CLOCK_MONOTONIC, &start);
    searchAll<TYPE>(A, n, queries, answers, numQueries, p);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double ms = ((end.tv_sec*1000000000. + end.tv_nsec) - (start.tv_sec*1000000000. + start.tv_nsec)) / 1000000.;   //millisecond

    #ifdef DEBUG
    bool correct = true;
    for (uint64_t i = 0; i < numQueries; i++) {
        if (answers[i] == n || A[answers[i]] != queries[i]) {
            //printf("query = %lu; found = %lu\n", queries[i], A[answers[i]]);
            correct = false;
        }
    }
    if (correct == false) printf("Searches failed!\n");
    else printf("Searches succeeded!\n");
    #endif

    free(queries);
    free(answers);

    return ms;
}
#endif