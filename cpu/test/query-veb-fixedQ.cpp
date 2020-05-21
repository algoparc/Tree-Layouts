#include "../src/veb-cycles.h"
#include "../src/query-veb.h"

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <number of queries> <number of threads>\n", argv[0]);
        exit(1);
    }

    uint64_t q = atol(argv[1]);
    if (q <= 0) {
        fprintf(stderr, "Number of queries must be a positive integer\n");
    }

    uint32_t p = atoi(argv[2]);
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
        timePermutevEB<uint64_t>(A, n, p);

        //Build table used in querying
        vEB_table *table = (vEB_table *)calloc(d, sizeof(vEB_table));
        buildTable(table, n, d, 0);
        #ifdef DEBUG
        printf("\n");
        for (int i = 0; i < d; ++i) {
            printf("i = %d; L = %lu; R = %lu; D = %u\n", i, table[i].L, table[i].R, table[i].D);
        }
        printf("\n");
        #endif

        //Querying
        for (uint32_t i = 0; i < ITERS; ++i) {
            time[i] = timeQueryvEB<uint64_t>(A, table, n, d, q, p);
        }
        printQueryTimings(n, q, time, p);

        free(A);
        free(table);
    }

    return 0;
}