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

#include "../src/bst-involutions.cuh"
#include "../src/query-bst.cuh"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <number of queries> <number of CPU threads>\n", argv[0]);
        exit(1);
    }

    int q = atol(argv[1]);
    if (q <= 0) {
        fprintf(stderr, "Number of queries must be a positive integer\n");
    }
    
    int p = atoi(argv[2]);
    omp_set_num_threads(p);

    double time[ITERS];

    for (int d = 22; d <= 31; ++d) {
        int n = pow(2, d) - 1;
        #ifdef DEBUG
        printf("n = 2^%d - 1 = %d\n", d, n);
        #endif

        int *A = (int *)malloc(n * sizeof(int));
        int *dev_A;
        cudaMalloc(&dev_A, n * sizeof(int));

        //Construction
        initSortedList<int>(A, n);
        cudaMemcpy(dev_A, A, n * sizeof(int), cudaMemcpyHostToDevice);
        timePermuteBST<int>(dev_A, n);
        cudaMemcpy(A, dev_A, n * sizeof(int), cudaMemcpyDeviceToHost);

        //Querying
        for (int i = 0; i < ITERS; ++i) {
            time[i] = timeQueryBST<int>(A, dev_A, n, q);
        }
        printQueryTimings(n, q, time); 

        free(A);
        cudaFree(dev_A);
    }

    return 0;
}