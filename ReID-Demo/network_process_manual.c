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

#include "param_layer_struct.h"
#include "network_process_manual.h"
#include "dnn_utils.h"
#include "ExtraKernels.h"

short int* l3_weights[NB_CONV];
int weights_size[NB_CONV];
short int* l3_bias[NB_CONV];
int bias_size[NB_CONV];
int __networ_init_done = 0;

typedef void (*ConvLayerFunctionType)(
                short int *,
                short int *,
                short int *,
                short int *,
                unsigned int
                );

ConvLayerFunctionType ConvLayerArray[NB_CONV] =
{
    ConvLayer0,
    ConvLayer1,
    ConvLayer2,
    ConvLayer3,
    ConvLayer4,
    ConvLayer5,
    ConvLayer6,
    ConvLayer7,
    ConvLayer8,
    ConvLayer9,
    ConvLayer10,
    ConvLayer11,
    ConvLayer12,
    ConvLayer13,
    ConvLayer14,
    ConvLayer15,
    ConvLayer16,
    ConvLayer17,
    ConvLayer18,
    ConvLayer19,
    ConvLayer20,
    ConvLayer21,
    ConvLayer22,
    ConvLayer23,
    ConvLayer24,
    ConvLayer25
};

// The function return L2 memory address where input image should be loader
// Expected format: 128x128xshort
short* network_init()
{
    L1_Memory =  pmsis_l1_malloc((_L1_Memory_SIZE>_ExtaKernels_L1_Memory_SIZE?_L1_Memory_SIZE:_ExtaKernels_L1_Memory_SIZE));
    if(L1_Memory == NULL)
    {
        PRINTF("L1 Working area alloc error\n");
        return NULL;
    }

    if(!__networ_init_done)
    {
        L2_Memory =  pmsis_l2_malloc(_L2_Memory_SIZE);
        if(L2_Memory == NULL)
        {
            PRINTF("L2 Working area alloc error\n");
            return NULL;
        }
        __networ_init_done = 1;
    }

    return memory_pool;
}

void network_deinit()
{
    pmsis_l1_malloc_free(L1_Memory, _L1_Memory_SIZE);
    pmsis_l2_malloc_free(L2_Memory,  _L2_Memory_SIZE);
    __networ_init_done = 0;
}

#define MAX(a, b) (((a)>(b))?(a):(b))

short* network_process(int* activation_size)
{
    short* layer_input;
    short* layer_output;
    //short* weight_base_address;
    //short* weights;
    //short* bias;

    layer_input = memory_pool;
    layer_output = memory_pool + MAX(get_activations_size(1), 128*128);
    //weight_base_address = layer_output + get_activations_size(0); // expects 3-channels 128x128

    //weights = weight_base_address;
    //bias = weights + weights_size[0] / sizeof(short);
    //bias = weight_base_address;
    //loadLayerFromL3ToL2(&hyper, l3_weights[0], weights, weights_size[0]);
    //loadLayerFromL3ToL2(&HyperRam, l3_bias[0], bias, bias_size[0]);

    ConvLayer0(layer_input, l3_weights[0], l3_bias[0], layer_output, convLayers[0].norm_data);

#ifdef STOP_AFTER_ConvLayer0
    *activation_size = get_activations_size(0);
    return layer_output;
#endif

    layer_input = layer_output;
    layer_output = memory_pool;
    //weight_base_address = layer_input + get_activations_size(0); // expects 3-channels 128x128


    //weights = weight_base_address;
    //bias = weights + weights_size[1] / sizeof(short);
    //bias = weight_base_address;

    //loadLayerFromL3ToL2(&hyper, l3_weights[1], weights, weights_size[1]);
    //loadLayerFromL3ToL2(&HyperRam, l3_bias[1], bias, bias_size[1]);

    ConvLayer1(layer_input, l3_weights[1], l3_bias[1], layer_output, convLayers[1].norm_data);

#ifdef STOP_AFTER_ConvLayer1
    *activation_size = get_activations_size(1);
    return layer_output;
#endif

    int previous_activation_size = get_activations_size(1);

    int fire_entry_idx = 2; // amount of layers before fire modules loop
    for(int i = 0; i < 3*8; i+=3)
    {
        // Fire module:
        // fire_entry_idx+i+0 - squeeze layer
        // fire_entry_idx+i+1 - e1x1
        // fire_entry_idx+i+2 - e3x3

        int concated_activation_size = get_activations_size(fire_entry_idx+i+1) +
                     get_activations_size(fire_entry_idx+i+2);

#ifdef NETWORK_DEBUG
        PRINTF("Fire module iteration %d\n", i/3);
        PRINTF("\tPrevious activation size: %d\n", previous_activation_size);
        PRINTF("\tConcatenated activation size: %d\n", concated_activation_size);
#endif
        if(i == 0)
        {
            // use output of previous convolutions
            layer_input = layer_output;
        }
        else
        {
            // always force input to E1x1 and E3x3 concatenation in the buffer beginning
            layer_input = memory_pool;
        }

        layer_output = memory_pool + MAX(previous_activation_size, concated_activation_size);

#ifdef NETWORK_DEBUG
        PRINTF("\tSqueeze Layer\n");
        PRINTF("\tSqueeze layer input offset %d\n", layer_input-memory_pool);
        PRINTF("\tSqueeze layer output offset %d\n", layer_output-memory_pool);
        PRINTF("\tActivation size: %d\n", get_activations_size(fire_entry_idx+i+0));
        PRINTF("\tWeight size, bytes: %d\n", weights_size[fire_entry_idx+i+0]);
        PRINTF("\tBias size, bytes: %d\n", bias_size[fire_entry_idx+i+0]);
#endif
        //weight_base_address = layer_output + get_activations_size(fire_entry_idx+i+0);
        //weights = weight_base_address;
        //bias = weight_base_address + weights_size[fire_entry_idx+i+0] / sizeof(short);
        //loadLayerFromL3ToL2(&hyper, l3_weights[fire_entry_idx+i+0], weights, weights_size[fire_entry_idx+i+0]);
        //bias = weight_base_address;
        //loadLayerFromL3ToL2(&HyperRam, l3_bias[fire_entry_idx+i+0], bias, bias_size[fire_entry_idx+i+0]);

        ConvLayerArray[fire_entry_idx+i+0](layer_input, l3_weights[fire_entry_idx+i+0], l3_bias[fire_entry_idx+i+0], layer_output,
                                           convLayers[fire_entry_idx+i+0].norm_data);

        layer_input = layer_output;
        layer_output = memory_pool;
        //weights = weight_base_address;
        //bias = weight_base_address + weights_size[fire_entry_idx+i+1] / sizeof(short);
        //bias = weight_base_address;

#ifdef NETWORK_DEBUG
        PRINTF("\tE1x1\n");
        PRINTF("\tActivation size: %d\n", get_activations_size(fire_entry_idx+i+1));
        PRINTF("\tWeight, bytes: %d\n", weights_size[fire_entry_idx+i+1]);
        PRINTF("\tBias size, bytes: %d\n", bias_size[fire_entry_idx+i+1]);
        PRINTF("\tE1x1 layer input offset %d\n", layer_input-memory_pool);
        PRINTF("\tE1x1 layer output offset %d\n", layer_output-memory_pool);
#endif
        //loadLayerFromL3ToL2(&hyper, l3_weights[fire_entry_idx+i+1], weights, weights_size[fire_entry_idx+i+1]);
        //loadLayerFromL3ToL2(&HyperRam, l3_bias[fire_entry_idx+i+1], bias, bias_size[fire_entry_idx+i+1]);

        ConvLayerArray[fire_entry_idx+i+1](layer_input, l3_weights[fire_entry_idx+i+1], l3_bias[fire_entry_idx+i+1], layer_output,
                                           convLayers[fire_entry_idx+i+1].norm_data);

        layer_output = memory_pool + get_activations_size(fire_entry_idx+i+1);
        //weights = weight_base_address;
        //bias = weight_base_address + weights_size[fire_entry_idx+i+2] / sizeof(short);
        //bias = weight_base_address;
#ifdef NETWORK_DEBUG
        PRINTF("\tE3x3\n");
        PRINTF("\tActivation size: %d\n", get_activations_size(fire_entry_idx+i+2));
        PRINTF("\tWeight size, bytes: %d\n", weights_size[fire_entry_idx+i+2]);
        PRINTF("\tBias size, bytes: %d\n", bias_size[fire_entry_idx+i+2]);
        PRINTF("\tE3x3 layer input offset %d\n", layer_input-memory_pool);
        PRINTF("\tE3x3 layer output offset %d\n", layer_output-memory_pool);
#endif

        //loadLayerFromL3ToL2(&hyper, l3_weights[fire_entry_idx+i+2], weights, weights_size[fire_entry_idx+i+2]);
        //loadLayerFromL3ToL2(&HyperRam, l3_bias[fire_entry_idx+i+2], bias, bias_size[fire_entry_idx+i+2]);
        ConvLayerArray[fire_entry_idx+i+2](layer_input, l3_weights[fire_entry_idx+i+2], l3_bias[fire_entry_idx+i+2], layer_output,
                                           convLayers[fire_entry_idx+i+2].norm_data);

        previous_activation_size = concated_activation_size;

#ifdef STOP_AFTER_FIRE_MODULE
        if(i == 3*STOP_AFTER_FIRE_MODULE)
        {
            *activation_size = concated_activation_size;
            return memory_pool;
        }
#endif
    }

    layer_input = memory_pool;
    layer_output = memory_pool + previous_activation_size;
    FinalAvgPool(layer_input, layer_output);
    *activation_size = 512;

    return layer_output;
}

void network_load(struct pi_device * fs)
{
    char buffer[64];
    for (unsigned int i = 0; i < NB_CONV; i++)
    {
        sprintf(buffer, "%s.weights.bin", convLayers[i].name);
        l3_weights[i] = loadLayerFromFsToL3(fs, buffer, &HyperRam, &weights_size[i]);
        sprintf(buffer, "%s.bias.bin", convLayers[i].name);
        l3_bias[i] = loadLayerFromFsToL3(fs, buffer, &HyperRam, &bias_size[i]);
    }
}

void network_free()
{
    for (unsigned int i = 0; i < NB_CONV; i++)
    {
        pi_ram_free(&HyperRam, (uint32_t)l3_weights[i], weights_size[i]);
        pi_ram_free(&HyperRam, (uint32_t)l3_bias[i], bias_size[i]);
    }
}
