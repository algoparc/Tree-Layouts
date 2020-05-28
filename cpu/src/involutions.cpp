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

#include "involutions.h"

//64-bit reversal
uint64_t reverse(uint64_t x) {
    x = (((x & 0xaaaaaaaaaaaaaaaa) >> 1) | ((x & 0x5555555555555555) << 1));
    x = (((x & 0xcccccccccccccccc) >> 2) | ((x & 0x3333333333333333) << 2));
    x = (((x & 0xf0f0f0f0f0f0f0f0) >> 4) | ((x & 0x0f0f0f0f0f0f0f0f) << 4));
    x = (((x & 0xff00ff00ff00ff00) >> 8) | ((x & 0x00ff00ff00ff00ff) << 8));
    x = (((x & 0xffff0000ffff0000) >> 16) | ((x & 0x0000ffff0000ffff) << 16));
    return((x >> 32) | (x << 32));
}

//reverse d least significant bits where the max value of i is d = log(i)
//base 2
uint64_t rev_d(uint64_t i, uint64_t d) {
    return reverse(i) >> (64-d);
}

//reverse b < d least significant bits
//base 2
uint64_t rev_b(uint64_t i, uint64_t d, uint64_t b) {
    uint64_t temp = i & (uint64_t)((((uint64_t)1) << b) - 1);
    temp = rev_d(temp, b);
    return temp + (i & (uint64_t)((((uint64_t)1) << d) - (((uint64_t)1) << b)));
}

//reverses d least significant digits in the given base, where d = log_base_(x)
uint64_t rev_base_d(uint64_t x, uint64_t base, uint64_t d) {
    uint64_t r = 0;
    uint64_t j = 1;
    uint64_t k = pow(base, d-1);
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

//reverses the b < d least significant digits in the given base, where d = log_base_(x)
uint64_t rev_base_b(uint64_t x, uint64_t base, uint64_t d, uint64_t b) {
    uint64_t r = 0;
    uint64_t k = pow(base, d-1);
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

//euclidean algorithm
uint64_t gcd(uint64_t a, uint64_t b) {
    uint64_t c;

    while (a != 0) {
        c = a;
        a = b % a; 
        b = c;
    }

    return b;
}

//extended euclidean algorithm
uint64_t egcd(uint64_t a, uint64_t b, int64_t *x, int64_t *y) {
    if (a == 0) {
        *x = 0;
        *y = 1;
        return b;
    }

    int64_t x1, y1;
    uint64_t g = egcd(b % a, a, &x1, &y1);

    *x = y1 - ((b/a) * x1);
    *y = x1;

    return g;
}

//involution used in shuffle
uint64_t involution(uint64_t r, uint64_t x, uint64_t m) {
    int64_t inverse, y;
    uint64_t g = gcd(x, m);

    egcd(x/g, m/g, &inverse, &y);

    if (inverse < 0) inverse += m/g;

    return (g * (r * inverse % (m/g)));
}