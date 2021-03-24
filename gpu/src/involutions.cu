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

#include "involutions.cuh"

//Reverse d least significant bits where the max value of i is d = log(i)
//base 2
__device__ uint64_t rev_d(uint64_t i, uint64_t d) {
  	return __brevll(i) >> (64-d);
}

//Reverse b < d least significant bits
//base 2
__device__ uint64_t rev_b(uint64_t i, uint64_t d, uint64_t b) {
  	uint64_t temp = i & (uint64_t)((((uint64_t)1) << b) - 1);
  	temp = rev_d(temp, b);
  	return temp + (i & (uint64_t)((((uint64_t)1) << d) - (((uint64_t)1) << b)));
}

//Reverses d least significant digits in the given base, where d = log_base_(x)
__device__ uint64_t rev_base_d(uint64_t x, uint64_t base, uint64_t d) {
    uint64_t r = 0;
    uint64_t j = 1;
    uint64_t k = ceil(pow(base, d-1));
    uint64_t digit;

    for (uint64_t i = d-1; i > 0; --i) {
        digit = x / k;

        r += digit * j;
        j *= base;

        x -= digit * k;
        k /= base;
    }
    r += x * j;

    return r;
}

//Reverses the b < d least significant digits in the given base, where d = log_base_(x)
__device__ uint64_t rev_base_b(uint64_t x, uint64_t base, uint64_t d, uint64_t b) {
    uint64_t r = 0;
    uint64_t k = ceil(pow(base, d-1));
    uint64_t digit;

    for (uint64_t i = d-1; i >= b; --i) {
        digit = x / k;
        r += digit * k;
        x -= digit * k;
        k /= base;
    }
    r += rev_base_d(x, base, b);

    return r;
}

//Euclidean algorithm
__device__ uint64_t gcd(uint64_t a, uint64_t b) {
    uint64_t c;

    while (a != 0) {
        c = a;
        a = b % a; 
        b = c;
    }

    return b;
}

//Extended Euclidean algorithm
__device__ uint64_t egcd(uint64_t a, uint64_t b) {
    uint64_t aa[2]={1,0}, bb[2]={0,1}, q;
    uint64_t result[3];

    while(1) {
        q = a / b; a = a % b;
        aa[0] = aa[0] - q*aa[1];  bb[0] = bb[0] - q*bb[1];
        if (a == 0) {
            result[0] = b; result[1] = aa[1]; result[2] = bb[1];
            break;
        }
        q = b / a; b = b % a;
        aa[1] = aa[1] - q*aa[0];  bb[1] = bb[1] - q*bb[0];
        if (b == 0) {
            result[0] = a; result[1] = aa[0]; result[2] = bb[0];
            break;
        }
    }
    return result[1];
}

//Involution used in shuffle_dk
__device__ uint64_t involution(uint64_t r, uint64_t x, uint64_t m) {
    uint64_t g = gcd(x, m);
    int64_t inverse = egcd(x/g, m/g);
    
    if (inverse < 0) inverse += m/g;

    return (g * (r * inverse % (m/g)));
}