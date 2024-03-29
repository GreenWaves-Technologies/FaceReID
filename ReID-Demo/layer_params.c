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

#include "layer_params.h"
#include "layer_params_quant.h"

const struct conv_layer_params convLayers[] =
{
    // Intro convolutions
    {.nb_if = 1,
     .nb_of = 3,
     .win = 128,
     .hin = 128,
     .kernel_width = 1,
     .kernel_height = 1,
     .relu = 0,
     .max_pool = 0,
     .pool_size = 0,
     .pool_stride = 0,
     .conv_padding = 1,
     .conv_stride = 1,
     .q.in = Q_IN_0,
     .q.out = Q_OUT_0,
     .q.weights = Q_WEIGHTS_0,
     .q.bias = Q_BIAS_0,
     .name = "Conv0MP0",
     .filename = "conv1.0"},

    {.nb_if = 3,
     .nb_of = 64,
     .win = 128,
     .hin = 128,
     .kernel_width = 3,
     .kernel_height = 3,
     .relu = 1,
     .max_pool = 1,
     .pool_size = 3,
     .pool_stride = 2,
     .conv_padding = 0,
     .conv_stride = 2,
     .q.in = Q_IN_1,
     .q.out = Q_OUT_1,
     .q.weights = Q_WEIGHTS_1,
     .q.bias = Q_BIAS_1,
     .name = "Conv1MP1",
     .filename = "features.0.0"},

    // Fire modules
    {.nb_if = 64,
     .nb_of = 16,
     .win = 31,
     .hin = 31,
     .kernel_width = 1,
     .kernel_height = 1,
     .relu = 1,
     .max_pool = 0,
     .pool_size = 0,
     .pool_stride = 0,
     .conv_padding = 1,
     .conv_stride = 1,
     .q.in = Q_IN_2,
     .q.out = Q_OUT_2,
     .q.weights = Q_WEIGHTS_2,
     .q.bias = Q_BIAS_2,
     .name = "Fire3_C1x1S",
     .filename = "features.3.squeeze.0"},

    {.nb_if = 16,
     .nb_of = 64,
     .win = 31,
     .hin = 31,
     .kernel_width = 1,
     .kernel_height = 1,
     .relu = 1,
     .max_pool = 0,
     .pool_size = 0,
     .pool_stride = 0,
     .conv_padding = 1,
     .conv_stride = 1,
     .q.in = Q_IN_3,
     .q.out = Q_OUT_3,
     .q.weights = Q_WEIGHTS_3,
     .q.bias = Q_BIAS_3,
     .name = "Fire3_C1x1",
     .filename = "features.3.expand1x1.0"},

    {.nb_if = 16,
     .nb_of = 64,
     .win = 31,
     .hin = 31,
     .kernel_width = 3,
     .kernel_height = 3,
     .relu = 1,
     .max_pool = 0,
     .pool_size = 0,
     .pool_stride = 0,
     .conv_padding = 1,
     .conv_stride = 1,
     .q.in = Q_IN_4,
     .q.out = Q_OUT_4,
     .q.weights = Q_WEIGHTS_4,
     .q.bias = Q_BIAS_4,
     .name = "Fire3_C3x3",
     .filename = "features.3.expand3x3.0"},

    {.nb_if = 128,
     .nb_of = 16,
     .win = 31,
     .hin = 31,
     .kernel_width = 1,
     .kernel_height = 1,
     .relu = 1,
     .max_pool = 0,
     .pool_size = 0,
     .pool_stride = 0,
     .conv_padding = 1,
     .conv_stride = 1,
     .q.in = Q_IN_5,
     .q.out = Q_OUT_5,
     .q.weights = Q_WEIGHTS_5,
     .q.bias = Q_BIAS_5,
     .name = "Fire4_C1x1S",
     .filename = "features.4.squeeze.0"},

    {.nb_if = 16,
     .nb_of = 64,
     .win = 31,
     .hin = 31,
     .kernel_width = 1,
     .kernel_height = 1,
     .relu = 1,
     .max_pool = 1,
     .pool_size = 3,
     .pool_stride = 2,
     .conv_padding = 1,
     .conv_stride = 1,
     .q.in = Q_IN_6,
     .q.out = Q_OUT_6,
     .q.weights = Q_WEIGHTS_6,
     .q.bias = Q_BIAS_6,
     .name = "Fire4_C1x1",
     .filename = "features.4.expand1x1.0"},

    {.nb_if = 16,
     .nb_of = 64,
     .win = 31,
     .hin = 31,
     .kernel_width = 3,
     .kernel_height = 3,
     .relu = 1,
     .max_pool = 1,
     .pool_size = 3,
     .pool_stride = 2,
     .conv_padding = 1,
     .conv_stride = 1,
     .q.in = Q_IN_7,
     .q.out = Q_OUT_7,
     .q.weights = Q_WEIGHTS_7,
     .q.bias = Q_BIAS_7,
     .name = "Fire4_C3x3",
     .filename = "features.4.expand3x3.0"},

    // MaxPool here

    {.nb_if = 128,
     .nb_of = 32,
     .win = 15,
     .hin = 15,
     .kernel_width = 1,
     .kernel_height = 1,
     .relu = 1,
     .max_pool = 0,
     .pool_size = 0,
     .pool_stride = 0,
     .conv_padding = 1,
     .conv_stride = 1,
     .q.in = Q_IN_8,
     .q.out = Q_OUT_8,
     .q.weights = Q_WEIGHTS_8,
     .q.bias = Q_BIAS_8,
     .name = "Fire6_C1x1S",
     .filename = "features.6.squeeze.0"},

    {.nb_if = 32,
     .nb_of = 128,
     .win = 15,
     .hin = 15,
     .kernel_width = 1,
     .kernel_height = 1,
     .relu = 1,
     .max_pool = 0,
     .pool_size = 0,
     .pool_stride = 0,
     .conv_padding = 1,
     .conv_stride = 1,
     .q.in = Q_IN_9,
     .q.out = Q_OUT_9,
     .q.weights = Q_WEIGHTS_9,
     .q.bias = Q_BIAS_9,
     .name = "Fire6_C1x1",
     .filename = "features.6.expand1x1.0"},

    {.nb_if = 32,
     .nb_of = 128,
     .win = 15,
     .hin = 15,
     .kernel_width = 3,
     .kernel_height = 3,
     .relu = 1,
     .max_pool = 0,
     .pool_size = 0,
     .pool_stride = 0,
     .conv_padding = 1,
     .conv_stride = 1,
     .q.in = Q_IN_10,
     .q.out = Q_OUT_10,
     .q.weights = Q_WEIGHTS_10,
     .q.bias = Q_BIAS_10,
     .name = "Fire6_C3x3",
     .filename = "features.6.expand3x3.0"},

    {.nb_if = 256,
     .nb_of = 32,
     .win = 15,
     .hin = 15,
     .kernel_width = 1,
     .kernel_height = 1,
     .relu = 1,
     .max_pool = 0,
     .pool_size = 0,
     .pool_stride = 0,
     .conv_padding = 1,
     .conv_stride = 1,
     .q.in = Q_IN_11,
     .q.out = Q_OUT_11,
     .q.weights = Q_WEIGHTS_11,
     .q.bias = Q_BIAS_11,
     .name = "Fire7_C1x1S",
     .filename = "features.7.squeeze.0"},

    {.nb_if = 32,
     .nb_of = 128,
     .win = 15,
     .hin = 15,
     .kernel_width = 1,
     .kernel_height = 1,
     .relu = 1,
     .max_pool = 1,
     .pool_size = 3,
     .pool_stride = 2,
     .conv_padding = 1,
     .conv_stride = 1,
     .q.in = Q_IN_12,
     .q.out = Q_OUT_12,
     .q.weights = Q_WEIGHTS_12,
     .q.bias = Q_BIAS_12,
     .name = "Fire7_C1x1",
     .filename = "features.7.expand1x1.0"},

    {.nb_if = 32,
     .nb_of = 128,
     .win = 15,
     .hin = 15,
     .kernel_width = 3,
     .kernel_height = 3,
     .relu = 1,
     .max_pool = 1,
     .pool_size = 3,
     .pool_stride = 2,
     .conv_padding = 1,
     .conv_stride = 1,
     .q.in = Q_IN_13,
     .q.out = Q_OUT_13,
     .q.weights = Q_WEIGHTS_13,
     .q.bias = Q_BIAS_13,
     .name = "Fire7_C3x3",
     .filename = "features.7.expand3x3.0"},

    // MaxPool here

    {256,48,7,7,1,1, 1, 0,0,0, 1,1, {Q_IN_14, Q_OUT_14, Q_WEIGHTS_14, Q_BIAS_14}, "Fire9_C1x1S", "features.9.squeeze.0"},
    {48,192,7,7,1,1, 1, 0,0,0, 1,1, {Q_IN_15, Q_OUT_15, Q_WEIGHTS_15, Q_BIAS_15}, "Fire9_C1x1",  "features.9.expand1x1.0"},
    {48,192,7,7,3,3, 1, 0,0,0, 1,1, {Q_IN_16, Q_OUT_16, Q_WEIGHTS_16, Q_BIAS_16}, "Fire9_C3x3",  "features.9.expand3x3.0"},

    {384,48,7,7,1,1, 1, 0,0,0, 1,1, {Q_IN_17, Q_OUT_17, Q_WEIGHTS_17, Q_BIAS_17}, "Fire10_C1x1S", "features.10.squeeze.0"},
    {48,192,7,7,1,1, 1, 0,0,0, 1,1, {Q_IN_18, Q_OUT_18, Q_WEIGHTS_18, Q_BIAS_18}, "Fire10_C1x1",  "features.10.expand1x1.0"},
    {48,192,7,7,3,3, 1, 0,0,0, 1,1, {Q_IN_19, Q_OUT_19, Q_WEIGHTS_19, Q_BIAS_19}, "Fire10_C3x3",  "features.10.expand3x3.0"},

    {384,64,7,7,1,1, 1, 0,0,0, 1,1, {Q_IN_20, Q_OUT_20, Q_WEIGHTS_20, Q_BIAS_20}, "Fire11_C1x1S", "features.11.squeeze.0"},
    {64,256,7,7,1,1, 1, 0,0,0, 1,1, {Q_IN_21, Q_OUT_21, Q_WEIGHTS_21, Q_BIAS_21}, "Fire11_C1x1",  "features.11.expand1x1.0"},
    {64,256,7,7,3,3, 1, 0,0,0, 1,1, {Q_IN_22, Q_OUT_22, Q_WEIGHTS_22, Q_BIAS_22}, "Fire11_C3x3",  "features.11.expand3x3.0"},

    {.nb_if = 512,
     .nb_of = 64,
     .win = 7,
     .hin = 7,
     .kernel_width = 1,
     .kernel_height = 1,
     .relu = 1,
     .max_pool = 0,
     .pool_size = 0,
     .pool_stride = 0,
     .conv_padding = 1,
     .conv_stride = 1,
     .q.in = Q_IN_23,
     .q.out = Q_OUT_23,
     .q.weights = Q_WEIGHTS_23,
     .q.bias = Q_BIAS_23,
     .name = "Fire12_C1x1S",
     .filename = "features.12.squeeze.0"},

    {.nb_if = 64,
     .nb_of = 256,
     .win = 7,
     .hin = 7,
     .kernel_width = 1,
     .kernel_height = 1,
     .relu = 1,
     .max_pool = 0,
     .pool_size = 0,
     .pool_stride = 0,
     .conv_padding = 1,
     .conv_stride = 1,
     .q.in = Q_IN_24,
     .q.out = Q_OUT_24,
     .q.weights = Q_WEIGHTS_24,
     .q.bias = Q_BIAS_24,
     .name = "Fire12_C1x1",
     .filename = "features.12.expand1x1.0"},

    {.nb_if = 64,
     .nb_of = 256,
     .win = 7,
     .hin = 7,
     .kernel_width = 3,
     .kernel_height = 3,
     .relu = 1,
     .max_pool = 0,
     .pool_size = 0,
     .pool_stride = 0,
     .conv_padding = 1,
     .conv_stride = 1,
     .q.in = Q_IN_25,
     .q.out = Q_OUT_25,
     .q.weights = Q_WEIGHTS_25,
     .q.bias = Q_BIAS_25,
     .name = "Fire12_C3x3",
     .filename = "features.12.expand3x3.0"},
};

int get_layer_in_size(int idx)
{
    if (idx >= NB_CONV)
        return 2 * get_layer_out_size(NB_CONV - 1);

    return convLayers[idx].nb_if * convLayers[idx].win * convLayers[idx].hin;
}

int get_layer_out_size(int idx)
{
    if (idx >= NB_CONV)
        return 512;

    int out_width = convLayers[idx].win;
    int out_height = convLayers[idx].hin;

    if (convLayers[idx].conv_padding == 0)
    {
        out_width -= convLayers[idx].kernel_width - 1;
        out_height -= convLayers[idx].kernel_height - 1;
    }

    out_width /= convLayers[idx].conv_stride;
    out_height /= convLayers[idx].conv_stride;

    // see output size formulae at https://pytorch.org/docs/0.4.0/nn.html#torch.nn.MaxPool2d
    // dilation = 1, padding = 0
    if (convLayers[idx].max_pool)
    {
        out_width = (out_width - (convLayers[idx].pool_size-1) - 1) / convLayers[idx].pool_stride + 1;
        out_height = (out_height - (convLayers[idx].pool_size-1) - 1) / convLayers[idx].pool_stride + 1;
    }

    int activation_size = convLayers[idx].nb_of * out_height * out_width;

//     PRINTF("Output size for layer %d: %dx%d\n", idx, out_width, out_height);
//     PRINTF("activation_size %d: %d\n", idx, activation_size);

    return activation_size;
}

int get_layer_weights_size(int idx)
{
    if (idx >= NB_CONV)
        return 0;

    return convLayers[idx].nb_if * convLayers[idx].nb_of * convLayers[idx].kernel_width * convLayers[idx].kernel_height;
}

int get_layer_bias_size(int idx)
{
    if (idx >= NB_CONV)
        return 0;

    return convLayers[idx].nb_of;
}
