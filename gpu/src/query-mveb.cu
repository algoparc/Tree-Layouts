/*
 * Copyright 2018-2020 Kyle Berney
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

#include "query-veb.cuh"

//Builds table of size d = log_2(n) + 1, which is used to query the vEB
//Described in "Cache Oblivious Search Trees via Binary Trees of Small Height" (Brodal, Fagerberg, and Jacob. SODA 2002)
//vEB_table: table of size d to store B, T, and D values
//n: number of nodes in the current tree
//d: total depth in the current tree, i.e., n = 2^d - 1
//root_depth: depth of the current tree in the full tree
void buildTable(vEB_table *table, uint64_t n, uint32_t d, uint32_t root_depth) {
    if (n == 1) return;

    uint32_t leaf_d = floor((d - 2) / 2.) + 1;

    float log_leaf = log2((float)leaf_d);

    if (log_leaf - ((int)log_leaf) != 0) {      //Not a perfectly balanced vEB
        leaf_d = pow(2, ceil(log_leaf));
    }

    uint32_t root_d = d - leaf_d;

    uint64_t root_n = pow(2, root_d) - 1;
    uint64_t leaf_n = pow(2, leaf_d) - 1;

    uint32_t i = d - leaf_d;
    table[i].L = leaf_n;         //size of the bottom/leaf trees
    table[i].R = root_n;         //size of the top/root tree
    table[i].D = root_depth;     //depth of the corresponding top/root tree

    buildTable(table, root_n, root_d, root_depth);                      //recurse on top/root tree
    buildTable(&table[i], leaf_n, leaf_d, i + root_depth);              //recurse on bottom/leaf tree
}