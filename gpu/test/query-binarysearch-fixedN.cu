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

#include "../src/binary-search.cuh"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <number of elements> <number of CPU threads>\n", argv[0]);
        exit(1);
    }

    uint64_t n = atol(argv[1]);
    int p = atoi(argv[2]);
    omp_set_num_threads(p);

    uint32_t *A = (uint32_t *)malloc(n * sizeof(uint32_t));
    uint32_t *dev_A;
    cudaMalloc(&dev_A, n * sizeof(uint32_t));

    double time[ITERS];

    initSortedList<uint32_t>(A, n);
    cudaMemcpy(dev_A, A, n * sizeof(uint32_t), cudaMemcpyHostToDevice);

    //Querying
    uint64_t q = 1000000;
    while (q <= 100000000) {
        for (uint32_t i = 0; i < ITERS; ++i) {
            time[i] = timeQuery<uint32_t>(A, dev_A, n, q);
        }
        printQueryTimings(n, q, time); 
        
        if (q == 1000000) {
            q = 10000000;
        }
        else {
            q += 10000000;
        }
    }

    free(A);
    cudaFree(dev_A);

    return 0;
}