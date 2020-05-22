#include "../src/bst-involutions.cuh"
#include "../src/query-bst.cuh"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr,"Usage: %s <integer d such that n = 2^d - 1> <number of CPU threads>\n", argv[0]);
        exit(1);
    }

    uint32_t d = atoi(argv[1]);
    uint64_t n = pow(2, d) - 1;
    #ifdef DEBUG
    printf("n = 2^%d - 1 = %lu\n", d, n);
    #endif

    int p = atoi(argv[2]);
    omp_set_num_threads(p);

    uint64_t *A = (uint64_t *)malloc(n * sizeof(uint64_t));
    uint64_t *dev_A;
    cudaMalloc(&dev_A, n * sizeof(uint64_t));

    double time[ITERS];

    //Construction
    initSortedList<uint64_t>(A, n);
    cudaMemcpy(dev_A, A, n * sizeof(uint64_t), cudaMemcpyHostToDevice);
    timePermuteBST<uint64_t>(dev_A, n, d, p);
    cudaMemcpy(A, dev_A, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    //Querying
    uint64_t q = 1000000;
    while (q <= 100000000) {
        for (uint32_t i = 0; i < ITERS; ++i) {
            time[i] = timeQueryBST<uint64_t>(A, dev_A, n, q);
        }
        printQueryTimings(n, q, time); 
        
        if (q == 1000000) {
            q = 10000000;
        }
        else {
            q += 10000000;
        }
    }

    free(A);
    cudaFree(dev_A);

    return 0;
}