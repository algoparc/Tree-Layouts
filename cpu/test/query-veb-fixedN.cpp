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

#include "../src/veb-cycles.h"
#include "../src/query-veb.h"

int main(int argc, char **argv) {
	if (argc != 3) {
		fprintf(stderr, "Usage: %s <number of elements> <number of threads>\n", argv[0]);
		exit(1);
	}

    uint64_t n = atol(argv[1]);
	uint32_t p = atoi(argv[2]);
	omp_set_num_threads(p);

    double time[ITERS];
	uint64_t *A = (uint64_t *)malloc(n * sizeof(uint64_t));

    //Construction
    initSortedList<uint64_t>(A, n);
    timePermutevEB<uint64_t>(A, n, p);

    uint32_t d = log2(n) + 1;
    if (n != pow(2, d) - 1) {     //non-perfect tree
        #ifdef DEBUG
        printf("\nQuerying: Non-perfect vEB\n");
        #endif

        uint32_t temp_d, root_d, leaf_d, inc_d;
        uint64_t temp_n, r, l, num_full, inc_n;
        temp_n = n;
        temp_d = d;

        uint32_t num_tables = 1;
        while (temp_d >= 2) {
            #ifdef DEBUG
            printf("\ntemp_n = %lu; temp_d = %u\n", temp_n, temp_d);
            #endif

            root_d = (temp_d - 2)/2 + 1;        //floor((d - 2)/2) + 1
            leaf_d = temp_d - root_d;           //ceil((d - 2)/2.) + 1

            r = pow(2, root_d) - 1;        //number of elements in the root subtree
            l = pow(2, leaf_d) - 1;        //number of elements in the full leaf subtrees

            #ifdef DEBUG
            printf("r = %lu; root_d = %u; l = %lu; leaf_d = %u\n", r, root_d, l, leaf_d);
            #endif

            num_full = (temp_n - r) / l;        //number of full leaf subtrees
            inc_n = temp_n - r - num_full*l;    //number of nodes in the incomplete leaf subtree

            #ifdef DEBUG
            printf("num_full = %lu; inc_n = %lu\n", num_full, inc_n);
            #endif

            if (inc_n == 0) break;

            inc_d = log2(inc_n) + 1;       //depth of the incomplete leaf subtree
            
            #ifdef DEBUG
            printf("inc_d = %u\n", inc_d);
            #endif

            ++num_tables;
            temp_n = inc_n;
            temp_d = inc_d;

            if (temp_n == pow(2, inc_d) - 1) break;
        }

        #ifdef DEBUG
        printf("\nnum_tables = %u\n", num_tables);
        #endif

        if (num_tables == 1) {
            vEB_table *table = (vEB_table *)calloc(d, sizeof(vEB_table));
            buildTable(table, n, d, 0);

            uint64_t q = 1000000;
            while (q <= 100000000) {
                for (uint32_t i = 0; i < ITERS; ++i) {
                    time[i] = timeQueryvEB<uint64_t>(A, table, n, d, q, p);
                }
                printQueryTimings(n, q, time, p);

                if (q == 1000000) {
                    q = 10000000;
                }
                else if (q >= 100000000) {
                    q += 100000000;
                }
                else {
                    q += 10000000;
                }
            }

            free(table);
        }
        else {
            vEB_table **tables = (vEB_table **)malloc(num_tables * sizeof(vEB_table *));
            tables[0] = (vEB_table *)calloc(d, sizeof(vEB_table));
            buildTable(tables[0], n, d, 0);

            #ifdef DEBUG
            printf("\nn = %lu; d = %u\n", n, d);
            for (uint32_t j = 0; j < d; ++j) {
                printf("depth = %u; L = %lu; R = %lu; D = %u\n", j, tables[0][j].L, tables[0][j].R, tables[0][j].D);
            }
            printf("\n");
            #endif

            uint64_t *idx = (uint64_t *)malloc(num_tables * sizeof(uint64_t));
            idx[num_tables - 1] = n;

            temp_n = n;
            temp_d = d;
            for (uint32_t i = 1; i < num_tables; ++i) {
                root_d = (temp_d - 2)/2 + 1;        //floor((d - 2)/2) + 1
                leaf_d = temp_d - root_d;           //ceil((d - 2)/2.) + 1

                r = pow(2, root_d) - 1;        //number of elements in the root subtree
                l = pow(2, leaf_d) - 1;        //number of elements in the full leaf subtrees

                num_full = (temp_n - r) / l;        //number of full leaf subtrees
                inc_n = temp_n - r - num_full*l;    //number of nodes in the incomplete leaf subtree
                inc_d = log2(inc_n) + 1;            //depth of the incomplete leaf subtree

                tables[i] = (vEB_table *)calloc(inc_d, sizeof(vEB_table));
                buildTable(tables[i], inc_n, inc_d, 0);

                #ifdef DEBUG
                printf("\nn = %lu; d = %u\n", inc_n, inc_d);
                for (uint32_t j = 0; j < inc_d; ++j) {
                    printf("depth = %u; L = %lu; R = %lu; D = %u\n", j, tables[i][j].L, tables[i][j].R, tables[i][j].D);
                }
                printf("\n");
                #endif

                idx[i-1] = n - inc_n;

                temp_n = inc_n;
                temp_d = inc_d;
            }

            #ifdef DEBUG
            printf("idx array: ");
            for (uint32_t i = 0; i < num_tables; ++i) {
                printf("%lu ", idx[i]);
            }
            printf("\n");
            #endif

            uint64_t q = 1000000;
            while (q <= 100000000) {
                for (uint32_t i = 0; i < ITERS; ++i) {
                    time[i] = timeQueryvEB_nonperfect<uint64_t>(A, tables, idx, num_tables, n, d, q, p);
                }
                printQueryTimings(n, q, time, p);

                if (q == 1000000) {
                    q = 10000000;
                }
                else if (q >= 100000000) {
                    q += 100000000;
                }
                else {
                    q += 10000000;
                }
            }

            for (uint32_t i = 0; i < num_tables; ++i) {
                free(tables[i]);
            }
            free(tables);
            free(idx);
        }
    }
    else {    //perfect tree
        #ifdef DEBUG
        printf("Querying: Perfect vEB\n");
        #endif

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
            else if (q >= 100000000) {
                q += 100000000;
            }
            else {
                q += 10000000;
            }
        }

        free(table);
    }

    free(A);

	return 0;
}