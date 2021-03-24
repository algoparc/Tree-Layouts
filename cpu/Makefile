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

OPTIONS=-O3 -fopenmp

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

veb-all: \
	build-veb-cycles.out \
	build-veb-involutions.out \
	query-veb-fixedN.out \
	query-veb-fixedQ.out

build-all: \
	cycles \
	build-bst-hybrid.out \
	involutions

cycles: \
	build-bst-cycles.out \
	build-btree-cycles.out \
	build-veb-cycles.out

involutions: \
	build-bst-involutions.out \
	build-btree-involutions.out \
	build-veb-involutions.out

query-all: \
	query-fixedN \
	query-fixedQ

query-fixedN: \
	query-binarysearch-fixedN.out \
	query-bst-fixedN.out \
	query-btree-fixedN.out \
	query-veb-fixedN.out

query-fixedQ: \
	query-binarysearch-fixedQ.out \
	query-bst-fixedQ.out \
	query-btree-fixedQ.out \
	query-veb-fixedQ.out

#Common stuff
src/common.o: src/common.cpp src/common.h src/params.h
	g++ $(OPTIONS) -c src/common.cpp -o src/common.o

src/involutions.o: src/involutions.cpp src/involutions.h src/params.h
	g++ $(OPTIONS) -c src/involutions.cpp -o src/involutions.o

#Binary Search
query-binarysearch-fixedN.out: test/query-binarysearch-fixedN.cpp src/binary-search.h src/common.o
	g++ $(OPTIONS) test/query-binarysearch-fixedN.cpp src/common.o -o query-binarysearch-fixedN.out

query-binarysearch-fixedQ.out: test/query-binarysearch-fixedQ.cpp src/binary-search.h src/common.o
	g++ $(OPTIONS) test/query-binarysearch-fixedQ.cpp src/common.o -o query-binarysearch-fixedQ.out

#BST
build-bst-cycles.out: test/build-bst-cycles.cpp src/bst-cycles.h src/cycles.h src/common.o
	g++ $(OPTIONS) test/build-bst-cycles.cpp src/common.o -o build-bst-cycles.out

build-bst-involutions.out: test/build-bst-involutions.cpp src/bst-involutions.h src/involutions.o src/common.o
	g++ $(OPTIONS) test/build-bst-involutions.cpp src/involutions.o src/common.o -o build-bst-involutions.out

build-bst-hybrid.out: test/build-bst-hybrid.cpp src/bst-hybrid.h src/cycles.h src/involutions.o src/common.o
	g++ $(OPTIONS) test/build-bst-hybrid.cpp src/involutions.o src/common.o -o build-bst-hybrid.out

query-bst-fixedN.out: test/query-bst-fixedN.cpp src/query-bst.h src/bst-hybrid.h src/cycles.h src/involutions.o src/common.o
	g++ $(OPTIONS) test/query-bst-fixedN.cpp src/involutions.o src/common.o -o query-bst-fixedN.out

query-bst-fixedQ.out: test/query-bst-fixedQ.cpp src/query-bst.h src/bst-hybrid.h src/cycles.h src/involutions.o src/common.o
	g++ $(OPTIONS) test/query-bst-fixedQ.cpp src/involutions.o src/common.o -o query-bst-fixedQ.out	

#Btree
build-btree-cycles.out: test/build-btree-cycles.cpp src/btree-cycles.h src/cycles.h src/common.o
	g++ $(OPTIONS) test/build-btree-cycles.cpp src/common.o -o build-btree-cycles.out

build-btree-involutions.out: test/build-btree-involutions.cpp src/btree-involutions.h src/involutions.o src/common.o
	g++ $(OPTIONS) test/build-btree-involutions.cpp src/involutions.o src/common.o -o build-btree-involutions.out

query-btree-fixedN.out: test/query-btree-fixedN.cpp src/query-btree.h src/btree-cycles.h src/cycles.h src/common.o
	g++ $(OPTIONS) test/query-btree-fixedN.cpp src/common.o -o query-btree-fixedN.out

query-btree-fixedQ.out: test/query-btree-fixedQ.cpp src/query-btree.h src/btree-cycles.h src/cycles.h src/common.o
	g++ $(OPTIONS) test/query-btree-fixedQ.cpp src/common.o -o query-btree-fixedQ.out

#vEB
src/query-veb.o: src/query-veb.cpp src/query-veb.h src/params.h
	g++ $(OPTIONS) -c src/query-veb.cpp -o src/query-veb.o

build-veb-cycles.out: test/build-veb-cycles.cpp src/veb-cycles.h src/cycles.h src/common.o
	g++ $(OPTIONS) test/build-veb-cycles.cpp src/common.o -o build-veb-cycles.out

build-veb-involutions.out: test/build-veb-involutions.cpp src/veb-involutions.h src/involutions.o src/common.o
	g++ $(OPTIONS) test/build-veb-involutions.cpp src/involutions.o src/common.o -o build-veb-involutions.out

query-veb-fixedN.out: test/query-veb-fixedN.cpp src/veb-cycles.h src/cycles.h src/query-veb.o src/common.o
	g++ $(OPTIONS) test/query-veb-fixedN.cpp src/query-veb.o src/common.o -o query-veb-fixedN.out

query-veb-fixedQ.out: test/query-veb-fixedQ.cpp src/veb-cycles.h src/cycles.h src/query-veb.o src/common.o
	g++ $(OPTIONS) test/query-veb-fixedQ.cpp src/query-veb.o src/common.o -o query-veb-fixedQ.out

clean:
	rm -f src/*.o
	rm -f *.out