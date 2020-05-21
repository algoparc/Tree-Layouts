#include "../src/veb-cycles.h"
#include "../src/query-veb.h"

int main(int argc, char **argv) {
	if (argc != 3) {
		fprintf(stderr, "Usage: %s <integer d such that n = 2^d - 1> <number of threads>\n", argv[0]);
		exit(1);
	}

	uint32_t d = atoi(argv[1]);
  	uint64_t n = pow(2, d) - 1;
  	#ifdef DEBUG
  	printf("n = 2^%d - 1 = %lu\n", d, n);
  	#endif

	uint32_t p = atoi(argv[2]);
	omp_set_num_threads(p);

    double time[ITERS];
	uint64_t *A = (uint64_t *)malloc(n * sizeof(uint64_t));

    //Construction
    initSortedList<uint64_t>(A, n);
    timePermutevEB<uint64_t>(A, n, p);

    //Build table used in querying
    vEB_table *table = (vEB_table *)calloc(d, sizeof(vEB_table));
    buildTable(table, n, d, 0);

    //Querying
    uint64_t q = 1000000;
    while (q <= 100000000) {
        for (uint32_t i = 0; i < ITERS; ++i) {
            time[i] = timeQueryvEB<uint64_t>(A, table, n, d, q, p);
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
    free(table);

	return 0;
}