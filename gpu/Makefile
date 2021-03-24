#
# Copyright 2018-2021 Kyle Berney
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

ARCH=\
    -gencode arch=compute_52,code=compute_52 \
    -gencode arch=compute_52,code=sm_52

FLAGS=-O3 -use_fast_math -lineinfo -Xcompiler -fopenmp

NVCC=nvcc

default: \
	build-all \
	query-all

bst-all: \
	build-bst-cycles.out \
	build-bst-involutions.out \
	build-bst-hybrid.out \
	query-bst-fixedN.out \
	query-bst-fixedQ.out

btree-all: \
	build-btree-cycles.out \
	build-btree-involutions.out \
	query-btree-fixedN.out \
	query-btree-fixedQ.out

mveb-all: \
	build-mveb-cycles.out \
	build-mveb-involutions.out \
	query-mveb-fixedN.out \
	query-mveb-fixedQ.out

build-all: \
	cycles \
	build-bst-hybrid.out \
	involutions

involutions: \
	build-bst-involutions.out \
	build-btree-involutions.out \
	build-mveb-involutions.out

cycles: \
	build-bst-cycles.out \
	build-btree-cycles.out \
	build-mveb-cycles.out

query-all: \
	query-fixedN \
	query-fixedQ

query-fixedN: \
	query-binarysearch-fixedN.out \
	query-bst-fixedN.out \
	query-btree-fixedN.out \
	query-mveb-fixedN.out

query-fixedQ: \
	query-binarysearch-fixedQ.out \
	query-bst-fixedQ.out \
	query-btree-fixedQ.out \
	query-mveb-fixedQ.out

#Common stuff
src/common.o: src/common.cu src/common.cuh src/params.cuh
	$(NVCC) $(ARCH) $(FLAGS) -dc src/common.cu -o src/common.o	

src/involutions.o: src/involutions.cu src/involutions.cuh src/params.cuh
	$(NVCC) $(ARCH) $(FLAGS) -dc src/involutions.cu -o src/involutions.o

#Binary Search
query-binarysearch-fixedN.out: test/query-binarysearch-fixedN.cu src/binary-search.cuh src/common.o
	$(NVCC) $(ARCH) $(FLAGS) test/query-binarysearch-fixedN.cu src/common.o -o query-binarysearch-fixedN.out

query-binarysearch-fixedQ.out: test/query-binarysearch-fixedQ.cu src/binary-search.cuh src/common.o
	$(NVCC) $(ARCH) $(FLAGS) test/query-binarysearch-fixedQ.cu src/common.o -o query-binarysearch-fixedQ.out	

#BST
test/build-bst-involutions.o: test/build-bst-involutions.cu src/bst-involutions.cuh src/params.cuh
	$(NVCC) $(ARCH) $(FLAGS) -dc test/build-bst-involutions.cu -o test/build-bst-involutions.o

test/build-bst-hybrid.o: test/build-bst-hybrid.cu src/bst-hybrid.cuh src/cycles.cuh src/params.cuh
	$(NVCC) $(ARCH) $(FLAGS) -dc test/build-bst-hybrid.cu -o test/build-bst-hybrid.o

test/query-bst-fixedN.o: test/query-bst-fixedN.cu src/query-bst.cuh src/bst-hybrid.cuh src/cycles.cuh src/params.cuh
	$(NVCC) $(ARCH) $(FLAGS) -dc test/query-bst-fixedN.cu -o test/query-bst-fixedN.o

test/query-bst-fixedQ.o: test/query-bst-fixedQ.cu src/query-bst.cuh src/bst-hybrid.cuh src/cycles.cuh src/params.cuh
	$(NVCC) $(ARCH) $(FLAGS) -dc test/query-bst-fixedQ.cu -o test/query-bst-fixedQ.o

build-bst-cycles.out: test/build-bst-cycles.cu src/btree-cycles.cuh src/cycles.cuh src/common.o
	$(NVCC) $(ARCH) $(FLAGS) test/build-bst-cycles.cu src/common.o -o build-bst-cycles.out

build-bst-involutions.out: test/build-bst-involutions.o src/involutions.o src/common.o
	$(NVCC) $(ARCH) $(FLAGS) test/build-bst-involutions.o src/involutions.o src/common.o -o build-bst-involutions.out

build-bst-hybrid.out: test/build-bst-hybrid.o src/involutions.o src/common.o
	$(NVCC) $(ARCH) $(FLAGS) test/build-bst-hybrid.o src/involutions.o src/common.o -o build-bst-hybrid.out

query-bst-fixedN.out: test/query-bst-fixedN.o src/involutions.o src/common.o
	$(NVCC) $(ARCH) $(FLAGS) test/query-bst-fixedN.o src/involutions.o src/common.o -o query-bst-fixedN.out

query-bst-fixedQ.out: test/query-bst-fixedQ.o src/involutions.o src/common.o
	$(NVCC) $(ARCH) $(FLAGS) test/query-bst-fixedQ.o src/involutions.o src/common.o -o query-bst-fixedQ.out

#Btree
test/build-btree-involutions.o: test/build-btree-involutions.cu src/btree-involutions.cuh src/params.cuh
	$(NVCC) $(ARCH) $(FLAGS) -dc test/build-btree-involutions.cu -o test/build-btree-involutions.o

build-btree-cycles.out: test/build-btree-cycles.cu src/btree-cycles.cuh src/cycles.cuh src/common.o
	$(NVCC) $(ARCH) $(FLAGS) test/build-btree-cycles.cu src/common.o -o build-btree-cycles.out

build-btree-involutions.out: test/build-btree-involutions.o src/involutions.o src/common.o
	$(NVCC) $(ARCH) $(FLAGS) test/build-btree-involutions.o src/involutions.o src/common.o -o build-btree-involutions.out

query-btree-fixedN.out: test/query-btree-fixedN.cu src/query-btree.cuh src/btree-cycles.cuh src/cycles.cuh src/common.o
	$(NVCC) $(ARCH) $(FLAGS) test/query-btree-fixedN.cu src/common.o -o query-btree-fixedN.out

query-btree-fixedQ.out: test/query-btree-fixedQ.cu src/query-btree.cuh src/btree-cycles.cuh src/cycles.cuh src/common.o
	$(NVCC) $(ARCH) $(FLAGS) test/query-btree-fixedQ.cu src/common.o -o query-btree-fixedQ.out

#mvEB
test/build-mveb-involutions.o: test/build-mveb-involutions.cu src/mveb-involutions.cuh src/params.cuh
	$(NVCC) $(ARCH) $(FLAGS) -dc test/build-mveb-involutions.cu -o test/build-mveb-involutions.o

src/query-mveb.o: src/query-mveb.cu src/query-mveb.cuh src/params.cuh
	$(NVCC) $(ARCH) $(FLAGS) -dc src/query-mveb.cu -o src/query-mveb.o

build-mveb-cycles.out: test/build-mveb-cycles.cu src/mveb-cycles.cuh src/cycles.cuh src/common.o
	$(NVCC) $(ARCH) $(FLAGS) test/build-mveb-cycles.cu src/common.o -o build-mveb-cycles.out

build-mveb-involutions.out: test/build-mveb-involutions.o src/involutions.o src/common.o
	$(NVCC) $(ARCH) $(FLAGS) test/build-mveb-involutions.o src/involutions.o src/common.o -o build-mveb-involutions.out

query-mveb-fixedN.out: test/query-mveb-fixedN.cu src/mveb-cycles.cuh src/cycles.cuh src/query-mveb.o src/common.o
	$(NVCC) $(ARCH) $(FLAGS) test/query-mveb-fixedN.cu src/query-mveb.o src/common.o -o query-mveb-fixedN.out

query-mveb-fixedQ.out: test/query-mveb-fixedQ.cu src/mveb-cycles.cuh src/cycles.cuh src/query-mveb.o src/common.o
	$(NVCC) $(ARCH) $(FLAGS) test/query-mveb-fixedQ.cu src/query-mveb.o src/common.o -o query-mveb-fixedQ.out

clean:
	rm -f src/*.o
	rm -f test/*.o
	rm -f *.out