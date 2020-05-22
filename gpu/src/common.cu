#include "common.cuh"

//debug: print out contents of array of size n
//assumes uint64_t data types
void printA(uint64_t *A, uint64_t n) {
  	uint64_t i;
  	for (i = 0; i < n; i++) {
      	printf("%lu ", A[i]);
    }
    printf("\n");
}

//Prints out timing metrics & its standard deviation across the ITERS runs
//Runtime (ms) & Throughput (n/us)
void printTimings(uint64_t n, double time[ITERS]) {
	double avg = 0.;		//average ms
    double std_dev = 0.;	//ms standard deviation
    double tp_avg = 0.;     //average throughput
    double tp_sd = 0.;      //throughput standard deviation

	for (uint32_t i = 0; i < ITERS; ++i) {
		avg += time[i];
	}
	avg /= ITERS;
    tp_avg = n/(1000. * avg);

    for (uint32_t i = 0; i < ITERS; ++i) {
        std_dev += (time[i] - avg) * (time[i] - avg);
        tp_sd += (n/(1000. * time[i]) - tp_avg) * (n/(1000. * time[i]) - tp_avg);
    }
    std_dev = sqrt(std_dev/ITERS);
    tp_sd = sqrt(tp_sd/ITERS);

    printf("%lu %lf %lf %lf %lf ", n, avg, std_dev, tp_avg, tp_sd);
    for (uint32_t i = 0; i < ITERS; ++i) {
        printf(" %lf", time[i]);
    }
    printf("\n");
}

//Prints out query timing metrics & its standard deviation across the ITERS runs
//Runtime (ms) & Throughput (q/us)
void printQueryTimings(uint64_t n, uint64_t q, double time[ITERS]) {
    double avg = 0.;        //average ms
    double std_dev = 0.;    //ms standard deviation
    double tp_avg = 0.;     //average throughput
    double tp_sd = 0.;      //throughput standard deviation

    for (uint32_t i = 0; i < ITERS; ++i) {
        avg += time[i];
    }
    avg /= ITERS;
    tp_avg = q/(1000. * avg);

    for (uint32_t i = 0; i < ITERS; ++i) {
        std_dev += (time[i] - avg) * (time[i] - avg);
        tp_sd += (q/(1000. * time[i]) - tp_avg) * (q/(1000. * time[i]) - tp_avg);
    }
    std_dev = sqrt(std_dev/ITERS);
    tp_sd = sqrt(tp_sd/ITERS);

    printf("%lu %lu %lf %lf %lf %lf ", n, q, avg, std_dev, tp_avg, tp_sd);
    for (uint32_t i = 0; i < ITERS; ++i) {
        printf(" %lf", time[i]);
    }
    printf("\n");
}