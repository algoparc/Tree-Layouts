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

#include "../src/bst-hybrid.cuh"
#include "../src/query-bst.cuh"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <number of queries> <number of CPU threads>\n", argv[0]);
        exit(1);
    }

    uint64_t q = atol(argv[1]);
    if (q <= 0) {
        fprintf(stderr, "Number of queries must be a positive integer\n");
    }
    
    int p = atoi(argv[2]);
    omp_set_num_threads(p);

    double time[ITERS];

    uint64_t n[13] = {
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
        2147483647
    };

    for (uint32_t i = 0; i < 13; ++i) {
        uint32_t *A = (uint32_t *)malloc(n[i] * sizeof(uint32_t));
        uint32_t *dev_A;
        cudaMalloc(&dev_A, n[i] * sizeof(uint32_t));

        //Construction
        initSortedList<uint32_t>(A, n[i]);
        cudaMemcpy(dev_A, A, n[i] * sizeof(uint32_t), cudaMemcpyHostToDevice);
        timePermuteBST<uint32_t>(dev_A, n[i]);
        cudaMemcpy(A, dev_A, n[i] * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        //Querying
        for (uint32_t j = 0; j < ITERS; ++j) {
            time[j] = timeQueryBST<uint32_t>(A, dev_A, n[i], q);
        }
        printQueryTimings(n[i], q, time); 

        free(A);
        cudaFree(dev_A);
    }

    return 0;
}