#ifndef QUERY_BST_H
#define QUERY_BST_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#include "params.h"
#include "common.h"

template<typename TYPE>
uint64_t searchBST(TYPE *A, uint64_t n, TYPE query) {
    uint64_t i = 0;

    while (i < n) {
        if (query == A[i]) {
          return i;
        }
        i = 2*i + 1 + (query > A[i]);
        __builtin_prefetch(&A[(8*i)+7], 0, 0);
    }

    return n;
}

//Performs all of the queries given in the array queries
//index in A of the queried items are saved in the answers array
template<typename TYPE>
void searchAll(TYPE *A, uint64_t n, TYPE *queries, uint64_t *answers, uint64_t numQueries, uint32_t p) {
    #pragma omp parallel for shared(A, n, queries, answers, numQueries, p) schedule(guided) num_threads(p)
    for (uint64_t i = 0; i < numQueries; ++i) {
        answers[i] = searchBST<TYPE>(A, n, queries[i]);
    }
}

//Generates numQueries random queries and returns the milliseconds needed to perform the queries on the given bst layout
template<typename TYPE>
double timeQueryBST(TYPE *A, uint64_t n, uint64_t numQueries, uint32_t p) {
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

template<typename TYPE>
uint64_t searchBST_noprefetch(TYPE *A, uint64_t n, TYPE query) {
    uint64_t i = 0;

    while (i < n) {
        if (query == A[i]) {
          return i;
        }
        i = 2*i + 1 + (query > A[i]);
    }

    return n;
}

//Performs all of the queries given in the array queries
//index in A of the queried items are saved in the answers array
template<typename TYPE>
void searchAll_noprefetch(TYPE *A, uint64_t n, TYPE *queries, uint64_t *answers, uint64_t numQueries, uint32_t p) {
    #pragma omp parallel for shared(A, n, queries, answers, numQueries, p) schedule(guided) num_threads(p)
    for (uint64_t i = 0; i < numQueries; ++i) {
        answers[i] = searchBST_noprefetch<TYPE>(A, n, queries[i]);
    }
}

//Generates numQueries random queries and returns the milliseconds needed to perform the queries on the given bst layout
template<typename TYPE>
double timeQueryBST_noprefetch(TYPE *A, uint64_t n, uint64_t numQueries, uint32_t p) {
  struct timespec start, end;

  TYPE *queries = createRandomQueries<TYPE>(A, n, numQueries);              //array to store random queries to perform
  uint64_t *answers = (uint64_t *)malloc(numQueries * sizeof(uint64_t));    //array to store the answers (i.e., index of the queried item)

  clock_gettime(CLOCK_MONOTONIC, &start);
  searchAll_noprefetch<TYPE>(A, n, queries, answers, numQueries, p);
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