#include "../src/bst-involutions.h"
#include "../src/query-bst.h"

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <number of queries> <number of threads> <0/1 ==> no prefetching/prefetching>\n", argv[0]);
        exit(1);
    }

    uint64_t q = atol(argv[1]);
    if (q <= 0) {
        fprintf(stderr, "Number of queries must be a positive integer\n");
    }

    int p = atoi(argv[2]);
    omp_set_num_threads(p);
    int prefetch = atoi(argv[3]);

    double time[ITERS];

    for (uint32_t d = 22; d <= 32; ++d) {
        uint64_t n = pow(2, d) - 1;
        #ifdef DEBUG
        printf("n = 2^%d - 1 = %lu\n", d, n);
        #endif

        uint64_t *A = (uint64_t *)malloc(n * sizeof(uint64_t));

        //Construction
        initSortedList<uint64_t>(A, n);
        timePermuteBST<uint64_t>(A, n, d, p);

        //Querying
        for (uint32_t i = 0; i < ITERS; ++i) {
            time[i] = (prefetch == 0) ? timeQueryBST_noprefetch<uint64_t>(A, n, q, p) : timeQueryBST<uint64_t>(A, n, q, p);
        }
        printQueryTimings(n, q, time, p);

        free(A);
    }

    return 0;
}