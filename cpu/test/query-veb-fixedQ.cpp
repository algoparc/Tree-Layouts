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

    uint64_t n[14] = {
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
        2147483647,
        4294967295
    };

    for (uint32_t j = 0; j < 14; ++j) {
    	uint64_t *A = (uint64_t *)malloc(n[j] * sizeof(uint64_t));

	    //Construction
	    initSortedList<uint64_t>(A, n[j]);
	    timePermutevEB<uint64_t>(A, n[j], p);

	    uint32_t d = log2(n[j]) + 1;

	    if (n[j] != pow(2, d) - 1) {     //non-perfect tree
	        uint32_t temp_d, root_d, leaf_d, inc_d;
	        uint64_t temp_n, r, l, num_full, inc_n;
	        temp_n = n[j];
	        temp_d = d;

	        uint32_t num_tables = 1;
	        while (temp_d >= 2) {
	            root_d = (temp_d - 2)/2 + 1;        //floor((d - 2)/2) + 1
	            leaf_d = temp_d - root_d;           //ceil((d - 2)/2.) + 1

	            r = pow(2, root_d) - 1;        		//number of elements in the root subtree
	            l = pow(2, leaf_d) - 1;        		//number of elements in the full leaf subtrees

	            num_full = (temp_n - r) / l;        //number of full leaf subtrees
	            inc_n = temp_n - r - num_full*l;    //number of nodes in the incomplete leaf subtree

	            if (inc_n == 0) break;

	            inc_d = log2(inc_n) + 1;       //depth of the incomplete leaf subtree

	            ++num_tables;
	            temp_n = inc_n;
	            temp_d = inc_d;

	            if (temp_n == pow(2, inc_d) - 1) break;
	        }

	        if (num_tables == 1) {
	            vEB_table *table = (vEB_table *)calloc(d, sizeof(vEB_table));
	            buildTable(table, n[j], d, 0);

	            for (uint32_t i = 0; i < ITERS; ++i) {
	            	time[i] = timeQueryvEB<uint64_t>(A, table, n[j], d, q, p);
	            }
	            printQueryTimings(n[j], q, time, p);

	            free(table);
	        }
	        else {
	            vEB_table **tables = (vEB_table **)malloc(num_tables * sizeof(vEB_table *));
	            tables[0] = (vEB_table *)calloc(d, sizeof(vEB_table));
	            buildTable(tables[0], n[j], d, 0);

	            uint64_t *idx = (uint64_t *)malloc(num_tables * sizeof(uint64_t));
	            idx[num_tables - 1] = n[j];

	            temp_n = n[j];
	            temp_d = d;
	            for (uint32_t i = 1; i < num_tables; ++i) {
	                root_d = (temp_d - 2)/2 + 1;        //floor((d - 2)/2) + 1
	                leaf_d = temp_d - root_d;           //ceil((d - 2)/2.) + 1

	                r = pow(2, root_d) - 1;        		//number of elements in the root subtree
	                l = pow(2, leaf_d) - 1;        		//number of elements in the full leaf subtrees

	                num_full = (temp_n - r) / l;        //number of full leaf subtrees
	                inc_n = temp_n - r - num_full*l;    //number of nodes in the incomplete leaf subtree
	                inc_d = log2(inc_n) + 1;            //depth of the incomplete leaf subtree

	                tables[i] = (vEB_table *)calloc(inc_d, sizeof(vEB_table));
	                buildTable(tables[i], inc_n, inc_d, 0);

	                idx[i-1] = n[j] - inc_n;

	                temp_n = inc_n;
	                temp_d = inc_d;
	            }

	            //Querying
	            for (uint32_t i = 0; i < ITERS; ++i) {
	                time[i] = timeQueryvEB_nonperfect<uint64_t>(A, tables, idx, num_tables, n[j], d, q, p);
	            }
	            printQueryTimings(n[j], q, time, p);

	            //Clean up
	            for (uint32_t i = 0; i < num_tables; ++i) {
	                free(tables[i]);
	            }
	            free(tables);
	            free(idx);
	        }
	    }
	    else {    //perfect tree
	        //Build table used in querying
	        vEB_table *table = (vEB_table *)calloc(d, sizeof(vEB_table));
	        buildTable(table, n[j], d, 0);

	        //Querying
            for (uint32_t i = 0; i < ITERS; ++i) {
                time[i] = timeQueryvEB<uint64_t>(A, table, n[j], d, q, p);
            }
            printQueryTimings(n[j], q, time, p);

	        free(table);
	    }

	    free(A);
    }

    return 0;
}