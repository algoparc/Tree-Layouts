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

#ifndef CYCLES_H
#define CYCLES_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include <math.h>

#include "params.h"

//Performs the equidistant gather on r root elements and l leaf elements
//Assumes r <= l
template<typename TYPE>
void equidistant_gather(TYPE *A, uint64_t r, uint64_t l) {
	for (uint64_t i = 0; i < r; ++i) {		//for each of the r cycles
		TYPE temp = A[(i+1)*(l+1) - 1];		//i-th root element
		for (uint64_t j = (i+1)*(l+1) - 1; j > i; j -= l) {
			A[j] = A[j-l];
		}
		A[i] = temp;
	}

	for (uint64_t i = 0; i < r; ++i) {		//leaf subtrees are 0-indexed
		//right shift of r - i or left shift of l - (r - i)
		//A[r + i*l] to A[r + (i+1)*l - 1]

		uint64_t kr = r - i;
		uint64_t kl = l - r + i;

		if (kr <= kl && kr != 0 && kr != l) {
			shift_right<TYPE>(&A[r + i*l], l, kr);
		}
		else if (kl != 0 && kl != l) {
			shift_left<TYPE>(&A[r + i*l], l, kl);
		}
	}
}

//Performs the equidistant gather on r root elements and l leaf elements using p threads
//Assumes r <= l
template<typename TYPE>
void equidistant_gather_parallel(TYPE *A, uint64_t r, uint64_t l, uint32_t p) {
	#pragma omp parallel for shared(A, r, l) schedule(guided, B) num_threads(p)
	for (uint64_t i = 0; i < r; ++i) {		//for each of the r cycles
		TYPE temp = A[(i+1)*(l+1) - 1];		//i-th root element
		for (uint64_t j = (i+1)*(l+1) - 1; j > i; j -= l) {
			A[j] = A[j-l];
		}
		A[i] = temp;
	}

	#pragma omp parallel for shared(A, r, l) schedule(guided) num_threads(p)
	for (uint64_t i = 0; i < r; ++i) {		//leaf subtrees are 0-indexed
		//right shift of r - i or left shift of l - (r - i)
		//A[r + i*l] to A[r + (i+1)*l - 1]

		uint64_t kr = r - i;
		uint64_t kl = l - r + i;

		if (kr <= kl && kr != 0 && kr != l) {
			shift_right<TYPE>(&A[r + i*l], l, kr);
		}
		else if (kl != 0 && kl != l) {
			shift_left<TYPE>(&A[r + i*l], l, kl);
		}
	}
}

//Performs the equidistant gather on r root elements and l leaf elements
//Executes min(r, B) cycles at once
//Assumes r <= l
template<typename TYPE>
void equidistant_gather_io(TYPE *A, uint64_t r, uint64_t l) {
	uint32_t b = (r < B) ? r : B;
	for (uint64_t i = 0; i < r; i += b) {		//perform b cycles simultaneously
		if (i > r - b) {		//last chunk may be incomplete
			b = r - (r/b)*b;
		}

		TYPE temp[b];

		//Load all root elements into temp
		for (uint64_t j = 0; j < b; ++j) {
			temp[j] = A[(i+j+1)*(l+1) - 1];		//(i+j)-th root element
		}

		//(b - 1) steps, not all cycles "participate"
		uint64_t idx = (i+b)*(l+1) - 1;
		for (uint64_t k = 1; k <= b - 1; ++k) {		
			for (uint64_t j = 0; j < k; ++j) {		//k cycles "participate"
				A[idx - j] = A[idx - j - l];
			}
			idx -= l;
		}

		//Remainder (i + 2) steps, all cycles "participate"
		for (; idx > i+b-1; idx -= l) {
			for (uint64_t j = 0; j < b; ++j) {
				A[idx - j] = A[idx - j - l];
			}
		}

		//Last step, write root elements to desired location
		for (uint64_t j = 0; j < b; ++j) {
			A[i + j] = temp[j];
		}
	}

	for (uint64_t i = 0; i < r; ++i) {		//leaf subtrees are 0-indexed
		//right shift of r - i or left shift of l - (r - i)
		//A[r + i*l] to A[r + (i+1)*l - 1]

		uint64_t kr = r - i;
		uint64_t kl = l - r + i;

		if (kr <= kl && kr != 0 && kr != l) {
			shift_right<TYPE>(&A[r + i*l], l, kr);
		}
		else if (kl != 0 && kl != l) {
			shift_left<TYPE>(&A[r + i*l], l, kl);
		}
	}
}

//Performs the equidistant gather on r root elements and l leaf elements using p threads
//Each thread executes min(r, B) cycles at once
//Assumes r <= l
template<typename TYPE>
void equidistant_gather_io_parallel(TYPE *A, uint64_t r, uint64_t l, uint32_t p) {
	#pragma omp parallel for shared(A, r, l, p) schedule(guided) num_threads(p)
	for (uint64_t i = 0; i < r; i += B) {		//perform b cycles simultaneously
		uint32_t b = (r < B) ? r : B;
		if (i > r - b) {		//last chunk may be incomplete
			b = r - (r/b)*b;
		}

		TYPE temp[b];

		//Load all root elements into temp
		for (uint64_t j = 0; j < b; ++j) {
			temp[j] = A[(i+j+1)*(l+1) - 1];		//(i+j)-th root element
		}

		//(b - 1) steps, not all cycles "participate"
		uint64_t idx = (i+b)*(l+1) - 1;
		for (uint64_t k = 1; k <= b - 1; ++k) {		
			for (uint64_t j = 0; j < k; ++j) {		//k cycles "participate"
				A[idx - j] = A[idx - j - l];
			}
			idx -= l;
		}

		//Remainder (i + 2) steps, all cycles "participate"
		for (; idx > i+b-1; idx -= l) {
			for (uint64_t j = 0; j < b; ++j) {
				A[idx - j] = A[idx - j - l];
			}
		}

		//Last step, write root elements to desired location
		for (uint64_t j = 0; j < b; ++j) {
			A[i + j] = temp[j];
		}
	}

	#pragma omp parallel for shared(A, r, l, p) schedule(guided) num_threads(p)
	for (uint64_t i = 0; i < r; ++i) {		//leaf subtrees are 0-indexed
		//right shift of r - i or left shift of l - (r - i)
		//A[r + i*l] to A[r + (i+1)*l - 1]

		uint64_t kr = r - i;
		uint64_t kl = l - r + i;

		if (kr <= kl && kr != 0 && kr != l) {
			shift_right<TYPE>(&A[r + i*l], l, kr);
		}
		else if (kl != 0 && kl != l) {
			shift_left<TYPE>(&A[r + i*l], l, kl);
		}
	}
}

//Performs the equidistant gather on m root elements and m leaf elements in chunks of size c
template<typename TYPE>
void equidistant_gather_chunks(TYPE *A, uint64_t m, uint64_t c) {	
	for (uint64_t i = 0; i < m; ++i) {
		uint32_t b = (B < c) ? B : c;
		TYPE temp[b];

		for (uint64_t j = 0; j < c; j += b) {
			if (j > c - b) {		//last chunk may be incomplete
				b = c - (c/b)*b;
			}

			for (uint32_t k = 0; k < b; ++k) {
				temp[k] = A[((i+1)*(m+1) - 1)*c + j + k];
			}

			for (uint64_t k = ((i+1)*(m+1) - 1)*c; k > i*c; k -= m*c) {
				for (uint32_t x = 0; x < b; ++x) {
					A[k + j + x] = A[k - m*c + j + x];
				}
			}

			for (uint32_t k = 0; k < b; ++k) {
				A[i*c + j + k] = temp[k];
			}
		}
	}

	for (uint64_t i = 0; i < m; ++i) {
		uint64_t kr = (m - i)*c;
		uint64_t kl = i*c;

		if (kr <= kl && kr != 0 && kr != m*c) {
			shift_right<TYPE>(&A[(i+1)*m*c], m*c, kr);
		}
		else if (kl != 0 && kl != m*c) {
			shift_left<TYPE>(&A[(i+1)*m*c], m*c, kl);
		}
	}
}

//Performs the equidistant gather on m root elements and m leaf elements in chunks of size c
//Assumes chunk size is larger than B
template<typename TYPE>
void equidistant_gather_chunks_parallel(TYPE *A, uint64_t m, uint64_t c, uint32_t p) {
	if (p <= m) {
		#pragma omp parallel for shared(A, m, c, p) schedule(guided) num_threads(p)
		for (uint64_t i = 0; i < m; ++i) {
			uint32_t b = (B < c) ? B : c;
			TYPE temp[b];

			for (uint64_t j = 0; j < c; j += b) {
				if (j > c - b) {		//last chunk may be incomplete
					b = c - (c/b)*b;
				}

				for (uint32_t k = 0; k < b; ++k) {
					temp[k] = A[((i+1)*(m+1) - 1)*c + j + k];
				}

				for (uint64_t k = ((i+1)*(m+1) - 1)*c; k > i*c; k -= m*c) {
					for (uint32_t x = 0; x < b; ++x) {
						A[k + j + x] = A[k - m*c + j + x];
					}
				}

				for (uint32_t k = 0; k < b; ++k) {
					A[i*c + j + k] = temp[k];
				}
			}
		}

		#pragma omp parallel for shared(A, m, c, p) schedule(guided) num_threads(p)
		for (uint64_t i = 0; i < m; ++i) {
			uint64_t kr = (m - i)*c;
			uint64_t kl = i*c;

			if (kr <= kl && kr != 0 && kr != m*c) {
				shift_right<TYPE>(&A[(i+1)*m*c], m*c, kr);
			}
			else if (kl != 0 && kl != m*c) {
				shift_left<TYPE>(&A[(i+1)*m*c], m*c, kl);
			}
		}
	}
	else {		//p > m; i.e., more processors than cycles
		uint32_t threads_per = ceil(p/(double)m);

		#pragma omp parallel for shared(A, m, c, p, threads_per) num_threads(m)
		for (uint64_t i = 0; i < m; ++i) {
			uint32_t b = (B < c) ? B : c;
			TYPE temp[b];

			uint32_t remainder = c % b;

			if (remainder == 0) {
				#pragma omp parallel for shared(A, m, c, p, threads_per, i, b) private(temp) schedule(guided) num_threads(threads_per)
				for (uint64_t j = 0; j < c; j += b) {
					for (uint32_t k = 0; k < b; ++k) {
						temp[k] = A[((i+1)*(m+1) - 1)*c + j + k];
					}
					
					for (uint64_t k = ((i+1)*(m+1) - 1)*c; k > i*c; k -= m*c) {
						for (uint32_t x = 0; x < b; ++x) {
							A[k + j + x] = A[k - m*c + j + x];
						}
					}

					for (uint32_t k = 0; k < b; ++k) {
						A[i*c + j + k] = temp[k];
					}
				}
			}
			else {
				#pragma omp parallel for shared(A, m, c, p, threads_per, i, b, remainder) private(temp) schedule(guided) num_threads(threads_per)
				for (uint64_t j = 0; j < c - remainder; j += b) {
					for (uint32_t k = 0; k < b; ++k) {
						temp[k] = A[((i+1)*(m+1) - 1)*c + j + k];
					}
					
					for (uint64_t k = ((i+1)*(m+1) - 1)*c; k > i*c; k -= m*c) {
						for (uint32_t x = 0; x < b; ++x) {
							A[k + j + x] = A[k - m*c + j + x];
						}
					}

					for (uint32_t k = 0; k < b; ++k) {
						A[i*c + j + k] = temp[k];
					}
				}

				//Last block is size remainder
				uint64_t j = c - remainder;
				for (uint32_t k = 0; k < remainder; ++k) {
					temp[k] = A[((i+1)*(m+1) - 1)*c + j + k];
				}
					
				for (uint64_t k = ((i+1)*(m+1) - 1)*c; k > i*c; k -= m*c) {
					for (uint32_t x = 0; x < remainder; ++x) {
						A[k + j + x] = A[k - m*c + j + x];
					}
				}

				for (uint32_t k = 0; k < remainder; ++k) {
					A[i*c + j + k] = temp[k];
				}
			}
		}

		#pragma omp parallel for shared(A, m, c, p) schedule(guided) num_threads(m)
		for (uint64_t i = 0; i < m; ++i) {
			uint64_t kr = (m - i)*c;
			uint64_t kl = i*c;

			if (kr <= kl && kr != 0 && kr != m*c) {
				shift_right_parallel<TYPE>(&A[(i+1)*m*c], m*c, kr, threads_per);
			}
			else if (kl != 0 && kl != m*c) {
				shift_left_parallel<TYPE>(&A[(i+1)*m*c], m*c, kl, threads_per);
			}
		}
	}
}

//Performs the extended equidistant gather for n = (b+1)^d - 1, where d is an arbitrary integer
template<typename TYPE>
void extended_equidistant_gather(TYPE *A, uint64_t n, uint64_t b) {
  	uint64_t m = n/(b+1);		//number of internal elements

  	if (m <= b) {		//base case: perform equidistant gather
    	equidistant_gather_io<TYPE>(A, m, m);
  	}
  	else {
    	//recurse on (b+1) partitions
    	for (uint64_t i = 0; i < b+1; ++i) {
     		extended_equidistant_gather<TYPE>(&A[i*m + i], m, b);
    	}

    	//merge partitions via equidistant gather of chunks of size c = ceil{m/(B+1)} on &A[c-1]
    	uint64_t c = ceil(m / (double)(b+1));
    	equidistant_gather_chunks<TYPE>(&A[c-1], b, c);
  	}
}

//Performs the extended equidistant gather for n = (b+1)^d - 1, where d is an arbitrary integer
template<typename TYPE>
void extended_equidistant_gather_parallel(TYPE *A, uint64_t n, uint64_t b, uint32_t p) {
  	uint64_t m = n/(b+1);		//number of internal elements

  	if (m <= b) {		//base case: perform equidistant gather
    	equidistant_gather_io_parallel<TYPE>(A, m, m, p);
  	}
  	else {
  		if (p <= b+1) {
  			//recurse on (b+1) partitions
  			#pragma omp parallel for shared(A, n, b, p) schedule(guided) num_threads(p)
	    	for (uint64_t i = 0; i < b+1; ++i) {
	     		extended_equidistant_gather<TYPE>(&A[i*m + i], m, b);
	    	}
  		}
  		else {
  			uint32_t threads_per = ceil(p/(double)(b+1));

  			//recurse on (b+1) partitions
  			#pragma omp parallel for shared(A, n, b, p) schedule(guided) num_threads(b+1)
	    	for (uint64_t i = 0; i < b+1; ++i) {
	     		extended_equidistant_gather_parallel<TYPE>(&A[i*m + i], m, b, threads_per);
	    	}
  		}

    	//merge partitions via equidistant gather of chunks of size c = ceil{m/(B+1)} on &A[c-1]
    	uint64_t c = ceil(m / (double)(b+1));
    	equidistant_gather_chunks_parallel<TYPE>(&A[c-1], b, c, p);
  	}
}

//Performs the extended equidistant gather for n = m(b+1), where m = n/(b+1)
template<typename TYPE>
void extended_equidistant_gather2(TYPE *A, uint64_t n, uint64_t b) {
	uint64_t m = n/(b+1);		//number of internal elements

	if (m <= b) {		//base case: perform equidistant gather
		equidistant_gather_io<TYPE>(A, m, b);
	}
	else {
		uint64_t r = m % (b+1);

		if (r == 0) {
			//recurse on (b+1) partitions
			for (uint64_t i = 0; i < b+1; ++i) {
				extended_equidistant_gather2<TYPE>(&A[i*m], m, b);
			}

			//merge partitions via equidistant gather of chunks of size c = m/(B+1) on &A[c]
			uint64_t c = m/(b+1);
			equidistant_gather_chunks<TYPE>(&A[c], (m - c)/c, c);
		}
		else {
			uint64_t size = r * (b+1);

			extended_equidistant_gather2<TYPE>(A, n - size, b);
			extended_equidistant_gather2<TYPE>(&A[n - size], size, b);
			shift_right<TYPE>(&A[m-r], (m-r)*b + r, r);
		}
	}
}

//Performs the extended equidistant gather for n = m(b+1), where m = n/(b+1)
template<typename TYPE>
void extended_equidistant_gather2_parallel(TYPE *A, uint64_t n, uint64_t b, uint32_t p) {
	uint64_t m = n/(b+1);		//number of internal elements

	if (m <= b) {		//base case: perform equidistant gather
		equidistant_gather_io_parallel<TYPE>(A, m, b, p);
	}
	else {
		uint64_t r = m % (b+1);

		if (r == 0) {
			if (p <= b+1) {
				//recurse on (b+1) partitions
				#pragma omp parallel for shared(A, n, b, p) schedule(guided) num_threads(p)
				for (uint64_t i = 0; i < b+1; ++i) {
					extended_equidistant_gather2<TYPE>(&A[i*m], m, b);
				}
			}
			else {
				uint32_t threads_per = ceil(p/(double)(b+1));

				//recurse on (b+1) partitions
				#pragma omp parallel for shared(A, n, b, p, threads_per) num_threads(b+1)
				for (uint64_t i = 0; i < b+1; ++i) {
					extended_equidistant_gather2_parallel<TYPE>(&A[i*m], m, b, threads_per);
				}
			}

			//merge partitions via equidistant gather of chunks of size c = m/(B+1) on &A[c]
			uint64_t c = m/(b+1);
			equidistant_gather_chunks_parallel<TYPE>(&A[c], (m - c)/c, c, p);
		}
		else {
			uint64_t size = r * (b+1);

			//Parallel Solution #1
			extended_equidistant_gather2_parallel<TYPE>(A, n - size, b, p);
			extended_equidistant_gather2_parallel<TYPE>(&A[n - size], size, b, p);

			/*//Parallel Solution #2
			if (p > 2) {
				#pragma omp parallel sections num_threads(2)
				{
					#pragma omp section
					{
						//extended_equidistant_gather2_parallel<TYPE>(A, n - size, b, ceil(p/2.));
						extended_equidistant_gather2_parallel<TYPE>(A, n - size, b, ceil((p/(double)n)*(n - size)));
					}
					#pragma omp section
					{
						//extended_equidistant_gather2_parallel<TYPE>(&A[n - size], size, b, ceil(p/2.));
						extended_equidistant_gather2_parallel<TYPE>(&A[n - size], size, b, ceil((p/(double)n)*size));
					}
				}
			}
			else {
				#pragma omp parallel sections num_threads(2)
				{
					#pragma omp section
					{
						extended_equidistant_gather2<TYPE>(A, n - size, b);
					}
					#pragma omp section
					{
						extended_equidistant_gather2<TYPE>(&A[n - size], size, b);
					}
				}
			}*/
			
			shift_right_parallel<TYPE>(&A[m-r], (m-r)*b + r, r, p);
		}
	}
}
#endif