#include "../src/bst-involutions.h"
#include "../src/query-bst.h"

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <integer d such that n = 2^d - 1> <number of threads> <0/1 ==> no prefetching/prefetching>\n", argv[0]);
        exit(1);
    }

    uint32_t d = atoi(argv[1]);
    uint64_t n = pow(2, d) - 1;
    #ifdef DEBUG
    printf("n = 2^%d - 1 = %lu\n", d, n);
    #endif

    int p = atoi(argv[2]);
    omp_set_num_threads(p);
    int prefetch = atoi(argv[3]);

    double time[ITERS];
    uint64_t *A = (uint64_t *)malloc(n * sizeof(uint64_t));

    //Construction
    initSortedList<uint64_t>(A, n);
    timePermuteBST<uint64_t>(A, n, d, p);

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
        else {
            q += 10000000;
        }
    }

    free(A);

    return 0;
}