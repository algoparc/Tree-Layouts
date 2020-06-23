/*
 * Copyright 2018-2020 Kyle Berney, Ben Karsin
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

#ifndef PARAMS_CUH
#define PARAMS_CUH

#define B 16		//B value for uint64_t
#define ITERS 10
//#define DEBUG 0  	//comment out to remove debug print statements
#define VERIFY		//comment out to remove query verification

#define BLOCKS 2048		//chosen arbitrarily
#define THREADS 256
#define WARPS 32		//number of threads per warp

#endif