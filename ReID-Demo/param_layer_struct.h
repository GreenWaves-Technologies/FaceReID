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

#ifndef __PARAM_LAYERS_STRUCT_H__
#define __PARAM_LAYERS_STRUCT_H__

struct param_conv_layer {
  int nb_if;
  int nb_of;
  int win;
  int hin;
  int kernel_width;
  int kernel_height;
  char relu;
  char max_pool;
  char pool_size;
  char pool_stride;
  char norm_data;
  char conv_padding;
  char conv_stride;
  char* name;
};

#define NB_CONV 26
extern struct param_conv_layer convLayers[NB_CONV];

#endif
