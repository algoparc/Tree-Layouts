#include "../src/btree-cycles.h"
#include "../src/query-btree.h"

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
    initSortedList<uint64_t>(A, n);
    timePermuteBtree<uint64_t>(A, n, b, p);

    //Querying
    uint64_t q = 1000000;
    while (q <= 100000000) {
        for (uint32_t i = 0; i < ITERS; ++i) {
            time[i] = timeQueryBtree<uint64_t>(A, n, b, q, p);
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