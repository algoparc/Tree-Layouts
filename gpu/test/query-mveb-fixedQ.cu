/*
 * Copyright 2018-2021 Kyle Berney
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 *    http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../src/mveb-cycles.cuh"
#include "../src/query-mveb.cuh"

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

    uint64_t n[13] = {
        4194303,
        8388607,
        10000000,
        16777215,
        33554431,
        67108863,
        100000000,
        134217727,
        268435455,
        536870911,
        1000000000,
        1073741823,
        2147483647
    };

    for (uint32_t j = 0; j < 13; ++j) {
        uint32_t *A = (uint32_t *)malloc(n[j] * sizeof(uint32_t));
        uint32_t *dev_A;
        cudaMalloc(&dev_A, n[j] * sizeof(uint32_t));

        //Construction
        initSortedList<uint32_t>(A, n[j]);
        cudaMemcpy(dev_A, A, n[j] * sizeof(uint32_t), cudaMemcpyHostToDevice);
        timePermutevEB<uint32_t>(dev_A, n[j]);
        cudaMemcpy(A, dev_A, n[j] * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        //Build table(s) used in querying
        uint32_t d = log2(n[j]) + 1;
        if (n[j] != pow(2, d) - 1) {     //non-perfect tree

            uint32_t temp_d, root_d, leaf_d, inc_d;
            uint64_t temp_n, r, l, num_full, inc_n;
            float log_leaf;
            temp_n = n[j];
            temp_d = d;

            uint32_t num_tables = 1;
            uint32_t table_size = d;
            while (temp_d >= 2) {
                root_d = (temp_d - 2)/2 + 1;        //floor((d - 2)/2) + 1
                leaf_d = temp_d - root_d;           //ceil((d - 2)/2.) + 1

                log_leaf = log2((float)leaf_d);
                if (log_leaf - ((int)log_leaf) != 0) {      //Not a perfectly balanced vEB, i.e., d is not a power of 2
                    leaf_d = pow(2, ceil(log_leaf));
                    root_d = temp_d - leaf_d;
                }

                r = pow(2, root_d) - 1;         //number of elements in the root subtree
                l = pow(2, leaf_d) - 1;         //number of elements in the full leaf subtrees

                num_full = (temp_n - r) / l;        //number of full leaf subtrees
                inc_n = temp_n - r - num_full*l;    //number of nodes in the incomplete leaf subtree

                if (inc_n == 0) break;

                inc_d = log2(inc_n) + 1;       //depth of the incomplete leaf subtree

                ++num_tables;
                table_size += inc_d;
                temp_n = inc_n;
                temp_d = inc_d;

                if (temp_n == pow(2, inc_d) - 1) break;
            }

            if (num_tables == 1) {
                vEB_table *table = (vEB_table *)calloc(d, sizeof(vEB_table));
                buildTable(table, n[j], d, 0);
                vEB_table *dev_table;
                cudaMalloc(&dev_table, d * sizeof(vEB_table));
                cudaMemcpy(dev_table, table, d * sizeof(vEB_table), cudaMemcpyHostToDevice);

                //Querying
                for (uint32_t i = 0; i < ITERS; ++i) {
                    time[i] = timeQueryvEB<uint32_t>(A, dev_A, dev_table, n[j], d, q);
                }
                printQueryTimings(n[j], q, time); 

                free(table);
                cudaFree(dev_table);
            }
            else {
                vEB_table *tables = (vEB_table *)calloc(table_size, sizeof(vEB_table));
                buildTable(tables, n[j], d, 0);

                uint64_t *idx = (uint64_t *)malloc(num_tables * sizeof(uint64_t));
                idx[num_tables - 1] = n[j];

                uint32_t *table_idx = (uint32_t *)malloc(num_tables * sizeof(uint32_t));
                table_idx[0] = 0;

                temp_n = n[j];
                temp_d = d;
                for (uint32_t i = 1; i < num_tables; ++i) {
                    table_idx[i] = table_idx[i-1] + temp_d;

                    root_d = (temp_d - 2)/2 + 1;        //floor((d - 2)/2) + 1
                    leaf_d = temp_d - root_d;           //ceil((d - 2)/2.) + 1

                    log_leaf = log2((float)leaf_d);
                    if (log_leaf - ((int)log_leaf) != 0) {      //Not a perfectly balanced vEB, i.e., d is not a power of 2
                        leaf_d = pow(2, ceil(log_leaf));
                        root_d = temp_d - leaf_d;
                    }

                    r = pow(2, root_d) - 1;         //number of elements in the root subtree
                    l = pow(2, leaf_d) - 1;         //number of elements in the full leaf subtrees

                    num_full = (temp_n - r) / l;        //number of full leaf subtrees
                    inc_n = temp_n - r - num_full*l;    //number of nodes in the incomplete leaf subtree
                    inc_d = log2(inc_n) + 1;            //depth of the incomplete leaf subtree

                    buildTable(&tables[table_idx[i]], inc_n, inc_d, 0);

                    idx[i-1] = n[j] - inc_n;

                    temp_n = inc_n;
                    temp_d = inc_d;
                }

                vEB_table *dev_tables;
                cudaMalloc(&dev_tables, table_size * sizeof(vEB_table));
                cudaMemcpy(dev_tables, tables, table_size * sizeof(vEB_table), cudaMemcpyHostToDevice);

                uint64_t *dev_idx;
                cudaMalloc(&dev_idx, num_tables * sizeof(uint64_t));
                cudaMemcpy(dev_idx, idx, num_tables * sizeof(uint64_t), cudaMemcpyHostToDevice);

                uint32_t *dev_table_idx;
                cudaMalloc(&dev_table_idx, num_tables * sizeof(uint32_t));
                cudaMemcpy(dev_table_idx, table_idx, num_tables * sizeof(uint32_t), cudaMemcpyHostToDevice);

                //Querying
                for (uint32_t i = 0; i < ITERS; ++i) {
                    time[i] = timeQueryvEB_nonperfect<uint32_t>(A, dev_A, dev_tables, dev_idx, dev_table_idx, n[j], d, num_tables, q);
                }
                printQueryTimings(n[j], q, time);

                free(tables);
                free(idx);
                free(table_idx);
                cudaFree(dev_tables);
                cudaFree(dev_idx);
                cudaFree(dev_table_idx);
            }
        }
        else {      //perfect tree
            vEB_table *table = (vEB_table *)calloc(d, sizeof(vEB_table));
            buildTable(table, n[j], d, 0);
            vEB_table *dev_table;
            cudaMalloc(&dev_table, d * sizeof(vEB_table));
            cudaMemcpy(dev_table, table, d * sizeof(vEB_table), cudaMemcpyHostToDevice);

            //Querying
            for (uint32_t i = 0; i < ITERS; ++i) {
                time[i] = timeQueryvEB<uint32_t>(A, dev_A, dev_table, n[j], d, q);
            }
            printQueryTimings(n[j], q, time);

            free(table);
            cudaFree(dev_table);
        }

        free(A);
        cudaFree(dev_A);
    }

    return 0;
}