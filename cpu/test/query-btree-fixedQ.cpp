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

#include "../src/btree-cycles.h"
#include "../src/query-btree.h"

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <number of queries> <b value> <number of threads>\n", argv[0]);
        exit(1);
    }

    uint64_t q = atol(argv[1]);
    if (q <= 0) {
        fprintf(stderr, "Number of queries must be a positive integer\n");
    }

    uint64_t b = atoi(argv[2]);
    uint64_t p = atoi(argv[3]);
    omp_set_num_threads(p);

    double time[ITERS];

    uint64_t n[14] = {
        4194303,
        8388607,
        10000000,
        16777215,
        33554431,
        67108863,
        100000000,
        134217727,
        268435455,
        536870911,
        1000000000,
        1073741823,
        2147483647,
        4294967295
    };

    for (uint32_t i = 0; i < 14; ++i) {
        uint64_t *A = (uint64_t *)malloc(n[i] * sizeof(uint64_t));
        
        //Construction
        initSortedList<uint64_t>(A, n[i]);
        timePermuteBtree<uint64_t>(A, n[i], b, p);

        //Querying
        for (uint32_t j = 0; j < ITERS; ++j) {
            time[j] = timeQueryBtree<uint64_t>(A, n[i], b, q, p);
        }
        printQueryTimings(n[i], q, time, p);

        free(A);
    }

    return 0;
}