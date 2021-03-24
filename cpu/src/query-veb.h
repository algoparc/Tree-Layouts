/*
 * Copyright 2018-2021 Kyle Berney, Ben Karsin
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

#ifndef QUERY_VEB_H
#define QUERY_VEB_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#include "params.h"
#include "common.h"

struct vEB_table {
	uint64_t L;			//size of the bottom/leaf tree
	uint64_t R;			//size of the corresponding top/root tree
	uint32_t D;			//depth of the root of the corresponding top/root tree
};
void buildTable(vEB_table *table, uint64_t n, uint32_t d, uint32_t root_depth);

//Searches given perfect vEB tree layout for the query'd element using the table composed of L, R, and D
//Assumes L, R, and D have been initialized via buildTable()
//pos is an array of size d which is used to store the 1-indexed position of the node visited during the query at current depth in the vEB tree
//Returns index of query'd element (if found)
//Otherwise, returns n (element not found)
template<typename TYPE>
uint64_t searchvEB(TYPE *A, vEB_table *table, uint64_t n, uint32_t d, TYPE query, uint64_t *pos, uint64_t tid) {
    TYPE current;

    uint32_t current_d = 0;
    uint64_t i = 1;         //1-indexed position of the current node in a BFS (i.e, level order) binary search tree
    pos[0] = 1;

    uint64_t index = 0;     //0-indexed position of the current node in the vEB tree
    while (index < n) {
        current = A[index];

        if (query == current) {
            return index;
        }

        i = 2*i + (query > current);
        current_d++;
        pos[current_d] = pos[table[current_d].D] + table[current_d].R + (i & table[current_d].R) * table[current_d].L;

        index = pos[current_d] - 1;
    }
    
    return n;
}

//Performs all of the queries given in the array queries
//index in A of the queried items are saved in the answers array
template<typename TYPE>
void searchAll(TYPE *A, vEB_table *table, uint64_t n, uint32_t d, TYPE *queries, uint64_t *answers, uint64_t numQueries, uint32_t p) {
    #pragma omp parallel shared(A, table, n, d, queries, answers, numQueries, p) num_threads(p)
    {
        uint64_t pos[d];
        uint64_t tid = omp_get_thread_num();

        for (uint64_t i = tid; i < numQueries; i += p) {
            answers[i] = searchvEB<TYPE>(A, table, n, d, queries[i], pos, tid);
        }
    }
}

//Generates numQueries random queries and returns the milliseconds needed to perform the queries on the given vEB tree layout
template<typename TYPE>
double timeQueryvEB(TYPE *A, vEB_table *table, uint64_t n, uint32_t d, uint64_t numQueries, uint32_t p) {
	struct timespec start, end;    
	
    TYPE *queries = createRandomQueries<TYPE>(A, n, numQueries);              //array to store random queries to perform
    uint64_t *answers = (uint64_t *)malloc(numQueries * sizeof(uint64_t));    //array to store the answers (i.e., index of the queried item)

    clock_gettime(CLOCK_MONOTONIC, &start);
    searchAll<TYPE>(A, table, n, d, queries, answers, numQueries, p);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double ms = ((end.tv_sec*1000000000. + end.tv_nsec) - (start.tv_sec*1000000000. + start.tv_nsec)) / 1000000.;		//millisecond

    #ifdef VERIFY
    bool correct = true;
    for (uint64_t i = 0; i < numQueries; i++) {
        if (answers[i] == n || A[answers[i]] != queries[i] || answers[i] == n) {
            #ifdef DEBUG
            printf("query = %lu; A[%lu] = %lu\n", queries[i], answers[i], A[answers[i]]);
            #endif
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

//Searches given non-perfect vEB layout for the query'd element using the tables composed of L, R, and D
//tables is an array of size num_tables pointing to each vEB table used for querying
    //E.g., tables[0] is used to query the full root and leaf subtrees
    //and tables[1] is used to query the first incomplete leaf subtree
    //Each subsequent entry in queries is used to query the next recursive incomplete leaf subtree
//idx is and array of size num_tables which gives the index of where the next incomplete leaf subtree starts
//Assumes all tables L, R, and D have been initialized via buildTable()
//pos is an array of size d which is used to store the 1-indexed position of the node visited during the query at current depth in the vEB tree
//Returns index of query'd element (if found)
//Otherwise, returns n (element not found)
template<typename TYPE>
uint64_t searchvEB_nonperfect(TYPE *A, vEB_table **tables, uint64_t *idx, uint32_t num_tables, TYPE query, uint64_t *pos, uint64_t tid) {
    TYPE current;
    uint32_t current_d;
    uint64_t i, index;

    TYPE *a = A;
    uint64_t offset = 0;
    uint32_t j = 0;

    while (j < num_tables) {
        current_d = 0;
        i = 1;         //1-indexed position of the current node in a BFS (i.e, level order) binary search tree
        pos[0] = 1;

        index = 0;     //0-indexed position of the current node in the vEB tree
        while (index + offset < idx[j]) {
            current = a[index];

            if (query == current) {
                return offset + index;
            }

            i = 2*i + (query > current);
            current_d++;
            pos[current_d] = pos[tables[j][current_d].D] + tables[j][current_d].R + (i & tables[j][current_d].R) * tables[j][current_d].L;

            index = pos[current_d] - 1;
        }

        a = &A[idx[j]];
        offset = idx[j];
        ++j;
    }

    return idx[num_tables-1];       //return n
}

//Performs all of the queries given in the array queries
//index in A of the queried items are saved in the answers array
template<typename TYPE>
void searchAll_nonperfect(TYPE *A, vEB_table **tables, uint64_t *idx, uint32_t num_tables, uint32_t d, TYPE *queries, uint64_t *answers, uint64_t numQueries, uint32_t p) {
    #pragma omp parallel shared(A, tables, idx, num_tables, d, queries, answers, numQueries, p) num_threads(p)
    {
        uint64_t pos[d];
        uint64_t tid = omp_get_thread_num();

        for (uint64_t i = tid; i < numQueries; i += p) {
            answers[i] = searchvEB_nonperfect<TYPE>(A, tables, idx, num_tables, queries[i], pos, tid);
        }
    }
}

//Generates numQueries random queries and returns the milliseconds needed to perform the queries on the given vEB tree layout
template<typename TYPE>
double timeQueryvEB_nonperfect(TYPE *A, vEB_table **tables, uint64_t *idx, uint32_t num_tables, uint64_t n, uint32_t d, uint64_t numQueries, uint32_t p) {
    struct timespec start, end;    
    
    TYPE *queries = createRandomQueries<TYPE>(A, n, numQueries);              //array to store random queries to perform
    uint64_t *answers = (uint64_t *)malloc(numQueries * sizeof(uint64_t));    //array to store the answers (i.e., index of the queried item)

    clock_gettime(CLOCK_MONOTONIC, &start);
    searchAll_nonperfect<TYPE>(A, tables, idx, num_tables, d, queries, answers, numQueries, p);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double ms = ((end.tv_sec*1000000000. + end.tv_nsec) - (start.tv_sec*1000000000. + start.tv_nsec)) / 1000000.;       //millisecond

    #ifdef VERIFY
    bool correct = true;
    for (uint64_t i = 0; i < numQueries; i++) {
        if (answers[i] == n || A[answers[i]] != queries[i] || answers[i] == n) {
            #ifdef DEBUG
            printf("query = %lu; A[%lu] = %lu\n", queries[i], answers[i], A[answers[i]]);
            #endif
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