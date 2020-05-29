# Tree-Layouts
Parallel in-place permutations between sorted order and the following implicit search tree layouts:
* Level-order Binary Search Tree (BST)
* Level-order B-tree (B-tree)
* van Emde Boas (vEB)
* Modified van Emde Boas (mvEB)

This library provides implementations for both the CPU (C++ with OpenMP) and the GPU (CUDA C++).

The algorithms implemented are described in the paper:

K. Berney, H. Casanova, A. Higuchi, B. Karsin, N. Sitchinava. "Beyond binary search: parallel in-place construction of implicit search tree layouts". In *Proceedings of the 32nd International Parallel and Distributed Processing Symposium (IPDPS)*, pages 1070-1079, 2018.

Additional details are available in:

K. Berney. ["Beyond binary search: parallel in-place construction of implicit search tree layouts" (Master's Thesis)](http://www2.hawaii.edu/~berneyk/publications/thesis.pdf). University of Hawaii at Manoa, 2018.




## Using Tree-Layouts

To compile all sample programs, type `make` (while in either the cpu or gpu directory).

In order to use any of the search tree layout permutations, include the corresponding header file (see the Organization section for a description of file names) and call its appropriate permutation function.

CPU:
* BST: `timePermuteBST<TYPE>(A, n, p)`
* B-tree: `timePermuteBtree<TYPE>(A, n, b, p)`
* vEB: `timePermutevEB<TYPE>(A, n, p)`
* mvEB: `timePermutemveB<TYPE>(A, n, p)`

GPU:
* BST: `timePermuteBST<TYPE>(dev_A, n)`
* B-tree: `timePermuteBtree<TYPE>(dev_A, n, b)`
* vEB: `timePermutevEB<TYPE>(dev_A, n)`
* mvEB: `timePermutemvEB<TYPE>(dev_A, n)`

To perform a set of queries on any of the resulting implicit search tree layouts, additionally include the corresponding query header file (see the Organization section for a description of file names) and call its appropriate query function.

CPU:
* BST: `timeQueryBST<TYPE>(A, n, q, p)`
* B-tree: `timeQueryBtree<TYPE>(A, n, b, q, p)`
* vEB: `timeQueryvEB<TYPE>(A, table, n, d, q, p)`
* mvEB: `timeQuerymvEB<TYPE>(A, table, n, d, q, p)`

GPU:
* BST: `timeQueryBST<TYPE>(A, dev_A, n, q)`
* B-tree: `timeQueryBtree<TYPE>(A, dev_A, n, b, q)`
* vEB: `timeQueryvEB<TYPE>(A, dev_A, dev_table, n, d, q)`
* mvEB: `timeQuerymvEB<TYPE>(A, dev_A, dev_table, n, d, q)`

The parameters of the above functions are defined as follows:
* `TYPE` is the data type of the elements
* `A` is a pointer to the start of the array of elements residing in CPU RAM space
* `dev_A` is a pointer to the start of the array of elements residing in GPU RAM space
* `n` is the number of elements in the array
* `p` is the number of CPU threads to use
* `table` is a pointer to the start of the data structure used to query the van Emde Boas tree layout and Modified van Emde Boas tree layout residing in CPU RAM space
* `dev_table` is a pointer to the start of the data structure used to query the van Emde Boas tree layout and Modified van Emde Boas tree layout residing in CPU RAM space
* `d` is the number of levels in the corresponding search tree

To initialize the table used in the van Emde Boas tree layout and Modified van Emde Boas tree layouts, execute the following code:
```
vEB_table *table = (vEB_table *)calloc(d, sizeof(vEB_table));
buildTable(table, n, d, 0);
```
(To build the table for the modified van Emde Boas tree layout, replace `vEB` with `mvEB`.)

To change the number of GPU threads launched and its organization into thread-blocks, change the parameters defined in `gpu/src/params.cuh`.
* `BLOCKS` is the number of thread-blocks to launch
* `THREADS` is the number of threads per thread-block

### Debug Mode

The `params.h` and `params.cuh` both contain a `#define DEBUG` statement, which should normally be commented out.
However, if the user wants to turn on debug mode, uncomment out this statement and recompile.
This will turn on various debug statements and verification checks throughout execution.




## Organization
The cpu and gpu directories each contain the necessary files needed to compile and run the library for the corresponding processing unit.
Both directories are organized similarly with a `Makefile`, a `src` folder, and a `test` folder.

For brevity, we describe the naming convention of files using the CPU file extensions (.cpp/.h) instead of the GPU file extensions (.cu/.cuh).
However, the descriptions apply to both the CPU and GPU files.

### src Directory
The src directory contains all the necessary code (.cpp/.h files for the CPU and .cu/.cuh files for the GPU) to perform all the various implicit search tree permutations.

* `src/params.h` contains all general compiler defined constants.
* `src/common.h` contains all general helper algorithms.
* `src/cycles.h` implements all cycle-leader helper algorithms.
* `src/involutions.h` and `src/involutions.cpp` implements all involution helper algorithms.
* `src/<search tree layout abbreviation>-cycles.h` implements the permutation for the corresponding search tree layout using the cycle-leader approach.
* `src/<search tree layout abbreviation>-involutions.h` implements the permutation for the corresponding search tree layout using the involutions approach.
* `src/query-<search tree layout abbreviation>.h` implements the permutation for the corresponding search tree layout using the involutions approach.
	* Note that for the van Emde Boas and Modified van Emde Boas tree layouts, a `src/query-<search tree layout abbreviation>.cpp` file is also needed.

### test Directory
The test directory contains all sample code for permuting and querying each of the implicit search tree layouts.

* `test/build-<search tree layout abbreviation>-cycles.cpp` performs the permutation for the corresponding search tree layout using the cycle-leader approach.
* `test/build-<search tree layout abbreviation>-involutions.cpp` performs the permutation for the corresponding search tree layout using the involutions approach.
* `test/query-<search tree layout abbreviation>-fixedN.cpp` queries the corresponding search tree layout, while varying Q and keeping N fixed.
* `test/query-<search tree layout abbreviation>-fixedQ.cpp` queries the corresponding search tree layout, while varying N and keeping Q fixed.




## Limitations and Future Work
* The current implementations for the BST, vEB, and mvEB tree layouts are limited to perfect search trees, i.e., for `n` being a power of two minus 1.
(The algorithms for performing these permutations on non-perfect search trees are described in the papers listed in the first section.)
* Asynchronous execution of GPU kernels can possibly be investigated and implemented.