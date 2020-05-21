#include "../src/binary-search.h"

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

    initSortedList<uint64_t>(A, n);

    //Querying
    uint64_t q = 1000000;
    while (q <= 100000000) {
        for (uint32_t i = 0; i < ITERS; ++i) {
            time[i] = timeQuery<uint64_t>(A, n, q, p);
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