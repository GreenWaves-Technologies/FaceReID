/*
 * Copyright 2019 GreenWaves Technologies, SAS
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

#ifndef NETWORK_PROCESS_H
#define NETWORK_PROCESS_H

#if !defined(__FREERTOS__)
# include "Gap.h"
#endif

#include "pmsis.h"

// The function return L2 memory address where input image should be loader
// Expected format: 128x128xshort
short *network_init(struct pi_device *cl, void *l2_buffer);
void network_deinit(struct pi_device *cl);

short *network_process(short *memory_pool, int *activation_size);

void network_load(struct pi_device * fs);
void network_free(void);

#endif
