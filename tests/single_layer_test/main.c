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

#ifndef CCN_PULP
  #include <stdio.h>
  #include <stdint.h>
#endif

#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "pmsis.h"
#include "bsp/fs.h"
#include "bsp/fs/readfs.h"
#include "bsp/fs/hostfs.h"
#include "bsp/flash/hyperflash.h"
#include "bsp/ram/hyperram.h"


#include "param_layer_struct.h"

#if defined(__FREERTOS__)
# include "GAPOC_BSP_Board_Init.h"
# include "pmsis_l2_malloc.h"
# include "pmsis_driver_core_api.h"
# include "pmsis_task.h"
# include "pmsis_os.h"
# include "drivers/hyperbus.h"
# include "hyperbus_cl_internal.h"
# include "pmsis_tiling.h"
#else
# include "Gap.h"
#endif

#include "CNN_BasicKernels.h"
#include "CnnKernels.h"

#include "dnn_utils.h"
#include "setup.h"

short* infer_result;
short * l2_x;
int activation_size = 0;

short int* l3_weights;
int weights_size;
short int* l3_bias;
int bias_size;

#ifndef TEST_LAYER_INDEX
# define TEST_LAYER_INDEX 0
#endif

int test_layer_idx = TEST_LAYER_INDEX;

typedef void (*ConvLayerFunctionType)(
                short int *,
                short int *,
                short int *,
                short int *,
                unsigned int,
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
    ConvLayer25,
};

void layer_load(struct pi_device * fs, int idx)
{
    if(idx < NB_CONV)
    {
        char buffer[64];
        sprintf(buffer, "%s.weights.bin", convLayers[idx].name);
        l3_weights = loadLayerFromFsToL3(fs, buffer, &HyperRam, &weights_size);
        sprintf(buffer, "%s.bias.bin", convLayers[idx].name);
        l3_bias = loadLayerFromFsToL3(fs, buffer, &HyperRam, &bias_size);
    }
    else
    {
        PRINTF("No weights required for final convolution\n");
    }
}

void layer_free()
{
    pi_hyperram_free(&HyperRam, l3_weights, weights_size);
    pi_hyperram_free(&HyperRam, l3_bias, bias_size);
}


// The function return L2 memory address where input image should be loader
// Expected format: 128x128xshort
short* layer_init()
{
    L1_Memory = pmsis_l1_malloc(_L1_Memory_SIZE);
    if(L1_Memory == NULL)
    {
        PRINTF("WorkingArea alloc error\n");
        return NULL;
    }

    L2_Memory =  pmsis_l2_malloc(_L2_Memory_SIZE);
    if(L2_Memory == NULL)
    {
        PRINTF("L2 Working area alloc error\n");
        return NULL;
    }

    return memory_pool;
}

#define MAX(a, b) (((a)>(b))?(a):(b))

short* layer_process(int layer_idx, int* activation_size)
{
    PRINTF("layer_process call\n");
    if(layer_idx < NB_CONV)
    {
        short* layer_input;
        short* layer_output;
        //short* weight_base_address;
        //short* weights;
        //short* bias;

        layer_input = memory_pool;
        layer_output = memory_pool + convLayers[layer_idx].nb_if*convLayers[layer_idx].win*convLayers[layer_idx].hin;
        //weight_base_address = layer_output + get_activations_size(layer_idx); // expects 3-channels 128x128

        //weights = weight_base_address;
        //bias = weights + weights_size / sizeof(short);

        //PRINTF("Loading weights\n");
        //loadLayerFromL3ToL2(&HyperRam, l3_weights, weights, weights_size);
        //loadLayerFromL3ToL2(&HyperRam, l3_bias, bias, bias_size);
        PRINTF("Convolution\n");
        ConvLayerArray[layer_idx](layer_input, l3_weights, l3_bias, layer_output, convLayers[layer_idx].norm_data, convLayers[layer_idx].norm_data);
        PRINTF("Convolution done\n");

        *activation_size = get_activations_size(layer_idx);
        return layer_output;
    }
    else
    {
        PRINTF("Global AvgPool Test\n");
        short* layer_input = memory_pool;
        short* layer_output = memory_pool + 2*get_activations_size(NB_CONV-1);
        *activation_size = 512;

        FinalAvgPool(layer_input, layer_output);

        return layer_output;
    }
}

static void cluster_main()
{
    PRINTF("cluster_main call\n");
    infer_result = layer_process(test_layer_idx, &activation_size);
}

void body(void *parameters)
{
    (void) parameters;
    struct pi_device cluster_dev;
    struct pi_cluster_conf cluster_conf;
    struct pi_cluster_task cluster_task;
    struct pi_hyperram_conf hyper_conf;

    PRINTF("main call\n");

    pi_hyperram_conf_init(&hyper_conf);
    pi_open_from_conf(&HyperRam, &hyper_conf);

    if (pi_ram_open(&HyperRam))
    {
        PRINTF("Error: cannot open Hyperram!\n");
        pmsis_exit(-2);
    }

    PRINTF("HyperRAM config done\n");

    // The hyper chip need to wait a bit.
    // TODO: find out need to wait how many times.
    pi_time_wait_us(1*1000*1000);

    PRINTF("Configuring Hyperflash and FS..");
    struct pi_device fs;
    struct pi_device flash;
    struct pi_readfs_conf conf;
    struct pi_hyperflash_conf flash_conf;
    pi_readfs_conf_init(&conf);

    pi_hyperflash_conf_init(&flash_conf);
    pi_open_from_conf(&flash, &flash_conf);

    if (pi_flash_open(&flash))
    {
        PRINTF("Error: Flash open failed\n");
        pmsis_exit(-3);
    }
    conf.fs.flash = &flash;

    pi_open_from_conf(&fs, &conf);

    int error = pi_fs_mount(&fs);
    if (error)
    {
        PRINTF("Error: FS mount failed with error %d\n", error);
        pmsis_exit(-3);
    }

    PRINTF("FS mounted\n");

    PRINTF("Loading layers to HyperRAM\n");
    layer_load(&fs, test_layer_idx);

    PRINTF("Unmount FS as it's not needed any more\n");
    pi_fs_unmount(&fs);
    PRINTF("FS unmounted\n");

    char *inputBlob = "../../../input.bin";
    char *outputBlob = "../../../output.bin";
    unsigned int Wi, Hi;

    PRINTF("Init cluster...\n");
    pi_cluster_conf_init(&cluster_conf);
    cluster_conf.id = 0;
    cluster_conf.device_type = 0;
    pi_open_from_conf(&cluster_dev, &cluster_conf);
    pi_cluster_open(&cluster_dev);
    PRINTF("Init cluster...done\n");

    #if !defined(__FREERTOS__)
    //Setting FC to 250MHz
    rt_freq_set(RT_FREQ_DOMAIN_FC, 250000000);

    //Setting Cluster to 150MHz
    // NOTE: Current Gap8 generation does not have clock divider for hyperbus
    // and using FC clocks over 150Mhz is dengerous
    rt_freq_set(RT_FREQ_DOMAIN_CL, 175000000);
    #endif

    l2_x = layer_init();
    PRINTF("Network init done\n");

    PRINTF("Reading input from host...\n");

    struct pi_hostfs_conf host_fs_conf;
    pi_hostfs_conf_init(&host_fs_conf);
    struct pi_device host_fs;

    pi_open_from_conf(&host_fs, &host_fs_conf);

    if (pi_fs_mount(&host_fs))
    {
        PRINTF("pi_fs_mount failed\n");
        pmsis_exit(-4);
    }
    void* host_file = pi_fs_open(&host_fs, inputBlob, PI_FS_FLAGS_READ);
    if (!host_file)
    {
        PRINTF("Failed to open file, %s\n", inputBlob);
        pmsis_exit(-5);
    }
    PRINTF("Host file open done\n");

    int input_size = 0;

    if(test_layer_idx < NB_CONV)
    {
        input_size = convLayers[test_layer_idx].nb_if*convLayers[test_layer_idx].win*convLayers[test_layer_idx].hin*sizeof(short);
    }
    else
    {
        input_size = 2*get_activations_size(NB_CONV-1)*sizeof(short);
    }

    PRINTF("input_size: %d\n", input_size);

    int read = pi_fs_read(host_file, l2_x, input_size);
    if(read != input_size)
    {
        PRINTF("Failed to read file\n");
        PRINTF("Expected input size %d, but read %d\n", input_size, read);
        pmsis_exit(-6);
    }

    pi_fs_close(host_file);

    PRINTF("Reading input from host...done\n");

    PRINTF("DNN inference..\n");
#ifdef PERF_COUNT
    unsigned int tm = rt_time_get_us();
#endif
    pi_cluster_send_task_to_cl(&cluster_dev, pi_cluster_task(&cluster_task, (void (*)(void *))cluster_main, NULL));
    PRINTF("DNN inference..done\n");

#ifdef PERF_COUNT
    tm = rt_time_get_us() - tm;
    PRINTF("DNN inference finished in %d microseconds\n", tm);
    PRINTF("Activations size, shorts: %d\n", activation_size);
#endif

    pi_cluster_close(&cluster_dev);

    host_file = pi_fs_open(&host_fs, outputBlob, PI_FS_FLAGS_WRITE);
    if (host_file == 0)
    {
        PRINTF("Failed to open host file, %s\n", outputBlob);
        pmsis_exit(-7);
    }

    pi_fs_write(host_file, infer_result, activation_size*sizeof(short));
    pi_fs_close(host_file);

    pi_fs_unmount(&host_fs);

    layer_free();

    pmsis_exit(0);
}

int main()
{
    PRINTF("Start Single Layer Test\n");
    pmsis_kickoff(body);
    return 0;
}
