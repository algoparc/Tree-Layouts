/*
 * Copyright 2018-2020 Kyle Berney
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

#include "../src/btree-cycles.cuh"
#include "../src/query-btree.cuh"

int main(int argc, char *argv[]){
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <number of queries> <b value> <number of CPU threads>\n", argv[0]);
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

    for (uint32_t d = 22; d <= 30; ++d) {
        uint64_t n = pow(2, d) - 1;
        #ifdef DEBUG
        printf("n = 2^%d - 1 = %lu; b = %lu\n", d, n, b);
        #endif

        uint64_t *A = (uint64_t *)malloc(n * sizeof(uint64_t));
        uint64_t *dev_A;
        cudaMalloc(&dev_A, n * sizeof(uint64_t));

        //Construction
        initSortedList<uint64_t>(A, n);
        cudaMemcpy(dev_A, A, n * sizeof(uint64_t), cudaMemcpyHostToDevice);
        timePermuteBtree<uint64_t>(dev_A, n, b);
        cudaMemcpy(A, dev_A, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);

        //Querying
        for (uint32_t i = 0; i < ITERS; ++i) {
            time[i] = timeQueryBtree<uint64_t>(A, dev_A, n, b, q);
        }
        printQueryTimings(n, q, time); 

        free(A);
        cudaFree(dev_A);
    }

    return 0;
}
