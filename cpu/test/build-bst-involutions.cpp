#include "../src/bst-involutions.h"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <integer d such that n = 2^d - 1> <number of threads>\n", argv[0]);
        exit(1);
    }

    uint32_t d = atoi(argv[1]);
    uint64_t n = pow(2, d) - 1;
    #ifdef DEBUG
    printf("n = 2^%d - 1 = %lu\n", d, n);
    #endif

    int p = atoi(argv[2]);
    omp_set_num_threads(p);

    double time[ITERS];
    uint64_t *A = (uint64_t *)malloc(n * sizeof(uint64_t));

    //Construction
    for (uint32_t i = 0; i < ITERS; ++i) {
        initSortedList<uint64_t>(A, n);
        time[i] = timePermuteBST<uint64_t>(A, n, d, p);
    }
    printTimings(n, time, p);

    free(A);

    return 0;
}