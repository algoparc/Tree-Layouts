#include "../src/btree-cycles.cuh"
#include "../src/query-btree.cuh"

int main(int argc, char *argv[]){
    if (argc != 4) {
        fprintf(stderr,"Usage: %s <number of elements> <b value> <number of CPU threads>\n", argv[0]);
        exit(1);
    }

    uint64_t n = atol(argv[1]);
    uint64_t b = atoi(argv[2]);
    #ifdef DEBUG
    printf("n = %lu; b = %lu\n", n, b);
    #endif

    uint64_t p = atoi(argv[3]);
    omp_set_num_threads(p);

    uint64_t *A = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *dev_A;
    cudaMalloc(&dev_A, n * sizeof(uint64_t));

    double time[ITERS];

    //Construction
    initSortedList<uint64_t>(A, n);
    cudaMemcpy(dev_A, A, n * sizeof(uint64_t), cudaMemcpyHostToDevice);
    timePermuteBtree<uint64_t>(dev_A, n, b);
    cudaMemcpy(A, dev_A, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    //Querying
    uint64_t q = 1000000;
    while (q <= 100000000) {
        for (uint32_t i = 0; i < ITERS; ++i) {
            time[i] = timeQueryBtree<uint64_t>(A, dev_A, n, b, q);
        }
        printTimings(q, time);

        if (q == 1000000) {
            q = 10000000;
        }
        else {
            q += 10000000;
        }
    }

    free(A);
    cudaFree(A);

    return 0;
}
