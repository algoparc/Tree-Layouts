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

#ifndef QUERY_BTREE_H
#define QUERY_BTREE_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#include "params.h"
#include "common.h"

//Searches given B-tree level order layout for the query'd element
//Returns index of query'd element (if found)
//Otherwise, returns n (element not found)
template<typename TYPE>
uint64_t searchBtree(TYPE *A, uint64_t n, uint64_t b, TYPE query) {
    TYPE current;
    uint64_t i = 0;
    uint64_t j;

    while (i < n) {
        for (j = 0; j < b; j++) {   //linear search to find child pointer to follow
            current = A[i + j];

            if (query == current) {
                return i+j;
            }
            else if (query < current) {
                break;
            }
        }
        i = (i + 1)*(b + 1) + j*b - 1;
    }

    return n;   //error value, i.e., element not found
}

//Performs all of the queries given in the array queries
//index in A of the queried items are saved in the answers array
template<typename TYPE>
void searchAll(TYPE *A, uint64_t n, uint64_t b, TYPE *queries, uint64_t *answers, uint64_t numQueries, uint32_t p) {
    #pragma omp parallel for shared(A, n, b, queries, answers, numQueries, p) schedule(guided) num_threads(p)
    for (uint64_t i = 0; i < numQueries; ++i) {
        answers[i] = searchBtree<TYPE>(A, n, b, queries[i]);
    }
}

//Generates numQueries random queries and returns the milliseconds needed to perform the queries on the given level-order btree layout
template<typename TYPE>
double timeQueryBtree(TYPE *A, uint64_t n, uint64_t b, uint64_t numQueries, uint32_t p) {
	struct timespec start, end;

	TYPE *queries = createRandomQueries<TYPE>(A, n, numQueries);              //array to store random queries to perform
	uint64_t *answers = (uint64_t *)malloc(numQueries * sizeof(uint64_t));    //array to store the answers (i.e., index of the queried item)

    clock_gettime(CLOCK_MONOTONIC, &start);
    searchAll<TYPE>(A, n, b, queries, answers, numQueries, p);
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