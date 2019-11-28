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

#ifndef __EXTRA_EMUL_STUBS_H__
#define __EXTRA_EMUL_STUBS_H__

#ifndef PPM_HEADER
# define PPM_HEADER                          40
#endif

#ifndef RT_FREQ_DOMAIN_FC
# define RT_FREQ_DOMAIN_FC 0
#endif

#ifndef RT_FREQ_DOMAIN_CL
# define RT_FREQ_DOMAIN_CL 0
#endif

#ifndef RT_PERF_CYCLES
# define RT_PERF_CYCLES 0
#endif

#ifdef __EMUL__

#include <stdlib.h>

typedef int rt_perf_t;

typedef enum {
  RT_ALLOC_FC_CODE,     /*!< Memory for fabric controller code. */
  RT_ALLOC_FC_DATA,     /*!< Memory for fabric controller data. */
  RT_ALLOC_FC_RET_DATA, /*!< Memory for fabric controller retentive data. */
  RT_ALLOC_CL_CODE,     /*!< Memory for cluster code. */
  RT_ALLOC_CL_DATA,  /*!< Memory for cluster data. */
  RT_ALLOC_L2_CL_DATA,  /*!< Memory for L2 cluster data. */
  RT_ALLOC_PERIPH,      /*!< Memory for peripherals data. */
} rt_alloc_e;

static int rt_bridge_open(const char* name, int flags, int mode, rt_event_t *event) {return 0;}
static int rt_bridge_read(int file, void* ptr, int len, rt_event_t *event) {return 0;}
static int rt_bridge_close(int file, rt_event_t *event) {return 0;}
static int rt_bridge_write(int file, void* ptr, int len, rt_event_t *event){return 0;}

static void rt_free(rt_alloc_e flags, void *chunk, int size) {free(chunk);}

static void rt_perf_init(rt_perf_t *perf){}
static void rt_perf_conf(rt_perf_t *perf, unsigned events){}
static int rt_freq_set(int domain, unsigned int freq) {return 0;}

static void rt_event_wait(rt_event_t *event){}
static rt_event_t *rt_event_get_blocking(rt_event_sched_t *sched){return NULL;}

#endif

#endif
