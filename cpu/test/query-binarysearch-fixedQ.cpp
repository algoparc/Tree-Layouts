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

#include "../src/binary-search.h"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <number of queries> <number of threads>\n", argv[0]);
        exit(1);
    }

    uint64_t q = atol(argv[1]);
    if (q <= 0) {
        fprintf(stderr, "Number of queries must be a positive integer\n");
    }

    int p = atoi(argv[2]);
    omp_set_num_threads(p);

    double time[ITERS];

    uint64_t n[45] = {
        4000000,
        4194303,
        8000000,
        8388607,
        10000000,
        15000000,
        16777215,
        20000000,
        30000000,
        33554431,
        40000000,
        50000000,
        60000000,
        67108863,
        70000000,
        80000000,
        90000000,
        100000000,
        110000000,
        120000000,
        130000000,
        134217727,
        140000000,
        150000000,
        160000000,
        170000000,
        180000000,
        190000000,
        200000000,
        268435455,
        300000000,
        400000000,
        500000000,
        536870911,
        600000000,
        700000000,
        800000000,
        900000000,
        1000000000,
        1073741823,
        2000000000,
        2147483647,
        3000000000,
        4000000000,
        4294967295
    };

    for (uint32_t i = 0; i < 45; ++i) {
        uint64_t *A = (uint64_t *)malloc(n[i] * sizeof(uint64_t));

        //Construction
        initSortedList<uint64_t>(A, n[i]);

        //Querying
        for (uint32_t j = 0; j < ITERS; ++j) {
            time[j] = timeQuery<uint64_t>(A, n[i], q, p);
        }
        printQueryTimings(n[i], q, time, p);

        free(A);
    }

    return 0;
}