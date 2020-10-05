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

#ifndef __LAYER_PARAMS_H__
#define __LAYER_PARAMS_H__

typedef struct {
    char in;
    char out;
    char weights;
    char bias;
} quant_t;

struct conv_layer_params {
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
  char conv_padding;
  char conv_stride;
  quant_t q;
  const char *name;
  const char *filename;
};

#define NB_CONV 26
extern const struct conv_layer_params convLayers[NB_CONV];

int get_layer_in_size(int idx);
int get_layer_out_size(int idx);
int get_layer_weights_size(int idx);
int get_layer_bias_size(int idx);

#endif
