#include "../src/btree-cycles.cuh"

int main(int argc, char *argv[]){
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <number of elements> <b value> <number of CPU threads>\n", argv[0]);
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
    for (uint32_t i = 0; i < ITERS; ++i) {
        initSortedList<uint64_t>(A, n);
        cudaMemcpy(dev_A, A, n * sizeof(uint64_t), cudaMemcpyHostToDevice);
        time[i] = timePermuteBtree<uint64_t>(dev_A, n, b);
        cudaMemcpy(A, dev_A, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    }
    printTimings(n, time);

    free(A);
    cudaFree(A);

    return 0;
}