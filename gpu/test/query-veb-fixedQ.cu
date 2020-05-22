#include "../src/veb-cycles.cuh"
#include "../src/query-veb.cuh"

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <number of queries> <number of CPU threads>\n", argv[0]);
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
        uint64_t *dev_A;
        cudaMalloc(&dev_A, n * sizeof(uint64_t));

        //Construction
        initSortedList<uint64_t>(A, n);
        cudaMemcpy(dev_A, A, n * sizeof(uint64_t), cudaMemcpyHostToDevice);
        timePermutevEB<uint64_t>(dev_A, n);
        cudaMemcpy(A, dev_A, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);

        //Build table used in querying
        vEB_table *table = (vEB_table *)calloc(d, sizeof(vEB_table));
        buildTable(table, n, d, 0);
        vEB_table *dev_table;
        cudaMalloc(&dev_table, d * sizeof(vEB_table));
        cudaMemcpy(dev_table, table, d * sizeof(vEB_table), cudaMemcpyHostToDevice);

        //Querying
        for (uint32_t i = 0; i < ITERS; ++i) {
            time[i] = timeQueryvEB<uint64_t>(A, dev_A, dev_table, n, d, q);
        }
        printQueryTimings(n, q, time); 

        free(A);
        free(table);
        cudaFree(dev_A);
        cudaFree(dev_table);
    }

    return 0;
}