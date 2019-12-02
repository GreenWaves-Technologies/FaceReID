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

#ifndef __FACEDET_PIPELINE_H__
#define __FACEDET_PIPELINE_H__

#include "cascade.h"

int check_detection_stability(cascade_reponse_t* hisotry, int history_size);
void detection_cluster_init(ArgCluster_T *ArgC);
void detection_cluster_main(ArgCluster_T *ArgC);

#endif