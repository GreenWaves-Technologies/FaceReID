/*
 * Copyright 2019-2020 GreenWaves Technologies, SAS
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "dnn_utils.h"
#include <ExtraKernels.h>
#include <CnnKernels.h>
#include "network_process.h"

int __network_init_done = 0;

#define MAX(a, b) (((a)>(b))?(a):(b))

// The function return L2 memory address where input image should be loader
// Expected format: 128x128xshort
short* network_init(struct pi_device *cl)
{
    ExtraKernels_L1_Memory = L1_Memory = pi_l1_malloc(cl, MAX(_L1_Memory_SIZE, _ExtraKernels_L1_Memory_SIZE));
    if(L1_Memory == NULL)
    {
        PRINTF("L1 Working area alloc error\n");
        return NULL;
    }

    if(!__network_init_done)
    {
        L2_Memory = (char *)memory_pool;
        SqueezeNetCNN_Construct();
        __network_init_done = 1;
    }
    return NetworkIn;
}

void network_deinit(struct pi_device *cl)
{
    SqueezeNetCNN_Destruct();
    pi_l1_free(cl, L1_Memory, _L1_Memory_SIZE);

    __network_init_done = 0;
}

short* network_process(int* activation_size)
{
    SqueezeNetCNN();
    *activation_size = 512;
    return NetworkOut;
}

void network_load(struct pi_device * fs)
{
    (void)fs;
    // Do nothing, graph code does it on the go
}

void network_free(void)
{
    // Do nothing
}
