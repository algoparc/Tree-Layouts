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

#include "../src/veb-cycles.cuh"

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <integer d such that n = 2^d - 1> <number of CPU threads>\n", argv[0]);
        exit(1);
    }

    uint32_t d = atoi(argv[1]);
    uint64_t n = pow(2, d) - 1;
    #ifdef DEBUG
    printf("n = 2^%d - 1 = %lu\n", d, n);
    #endif

    uint32_t p = atoi(argv[2]);
    omp_set_num_threads(p);

    uint64_t *A = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *dev_A;
    cudaMalloc(&dev_A, n * sizeof(uint64_t));

    double time[ITERS];

    //Construction
    for (uint32_t i = 0; i < ITERS; ++i) {
        initSortedList<uint64_t>(A, n);
        cudaMemcpy(dev_A, A, n * sizeof(uint64_t), cudaMemcpyHostToDevice);
        time[i] = timePermutevEB<uint64_t>(dev_A, n);
        cudaMemcpy(A, dev_A, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    }
    printTimings(n, time);

    free(A);
    cudaFree(A);

    return 0;
}