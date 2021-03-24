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

#include "../src/bst-hybrid.h"
#include "../src/query-bst.h"

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <number of elements> <number of threads> <0/1 ==> no prefetching/prefetching>\n", argv[0]);
        exit(1);
    }

    int n = atol(argv[1]);
    int p = atoi(argv[2]);
    omp_set_num_threads(p);
    int prefetch = atoi(argv[3]);

    double time[ITERS];
    uint64_t *A = (uint64_t *)malloc(n * sizeof(uint64_t));

    //Construction
    initSortedList<uint64_t>(A, n);
    timePermuteBST<uint64_t>(A, n, p);

    //Querying
    uint64_t q = 1000000;
    while (q <= 100000000) {
        for (uint32_t i = 0; i < ITERS; ++i) {
            time[i] = (prefetch == 0) ? timeQueryBST_noprefetch<uint64_t>(A, n, q, p) : timeQueryBST<uint64_t>(A, n, q, p);
        }
        printQueryTimings(n, q, time, p);

        if (q == 1000000) {
            q = 10000000;
        }
        else if (q >= 100000000) {
            q += 100000000;
        }
        else {
            q += 10000000;
        }
    }

    free(A);

    return 0;
}