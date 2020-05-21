#include "../src/btree-cycles.h"

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <integer n> <b value> <number of threads>\n", argv[0]);
        exit(1);
    }

    uint64_t n = atol(argv[1]);
    uint64_t b = atoi(argv[2]);
    uint64_t p = atoi(argv[3]);
    omp_set_num_threads(p);

    double time[ITERS];
    uint64_t *A = (uint64_t *)malloc(n * sizeof(uint64_t));

    //Construction
    for (uint32_t i = 0; i < ITERS; ++i) {
        initSortedList<uint64_t>(A, n);
        time[i] = timePermuteBtree<uint64_t>(A, n, b, p);
    }
    printTimings(n, time, p);

    free(A);

    return 0;
}