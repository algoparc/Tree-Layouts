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

    for (uint32_t d = 22; d <= 32; ++d) {
        uint64_t n = pow(2, d) - 1;
        #ifdef DEBUG
        printf("n = 2^%d - 1 = %lu\n", d, n);
        #endif

        uint64_t *A = (uint64_t *)malloc(n * sizeof(uint64_t));

        //Construction
        initSortedList<uint64_t>(A, n);

        //Querying
        for (uint32_t i = 0; i < ITERS; ++i) {
            time[i] = timeQuery<uint64_t>(A, n, q, p);
        }
        printQueryTimings(n, q, time, p);

        free(A);
    }

    return 0;
}