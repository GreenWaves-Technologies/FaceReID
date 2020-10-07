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

#if defined(__FREERTOS__)
# include "GAPOC_BSP_Board_Init.h"
# include "pmsis_driver_core_api.h"
# include "pmsis_task.h"
# include "pmsis_os.h"
# include "drivers/hyperbus.h"
# include "hyperbus_cl_internal.h"
# include "pmsis_tiling.h"
#else
# include "Gap.h"
#endif

#include "layer_params.h"

#include "CNN_BasicKernels.h"
#include "CnnKernels.h"

#include "dnn_utils.h"
#include "setup.h"

short* infer_result;
short * l2_x;

short int* l3_weights;
int weights_size;
short int* l3_bias;
int bias_size;

int test_layer_idx = TEST_LAYER_INDEX;

typedef void (*ConvLayerFunctionType)(short int *, short int *, short int *, short int *);

ConvLayerFunctionType ConvLayerArray[NB_CONV] =
{
    Conv0MP0,
    Conv1MP1,
    Fire3_C1x1S,
    Fire3_C1x1,
    Fire3_C3x3,
    Fire4_C1x1S,
    Fire4_C1x1,
    Fire4_C3x3,
    Fire6_C1x1S,
    Fire6_C1x1,
    Fire6_C3x3,
    Fire7_C1x1S,
    Fire7_C1x1,
    Fire7_C3x3,
    Fire9_C1x1S,
    Fire9_C1x1,
    Fire9_C3x3,
    Fire10_C1x1S,
    Fire10_C1x1,
    Fire10_C3x3,
    Fire11_C1x1S,
    Fire11_C1x1,
    Fire11_C3x3,
    Fire12_C1x1S,
    Fire12_C1x1,
    Fire12_C3x3
};

void layer_load(struct pi_device * fs, int idx)
{
    if(idx < NB_CONV)
    {
        char buffer[64];
        sprintf(buffer, "%s.weights.bin", convLayers[idx].filename);
        l3_weights = loadLayerFromFsToL3(fs, buffer, &HyperRam, &weights_size);
        sprintf(buffer, "%s.bias.bin", convLayers[idx].filename);
        l3_bias = loadLayerFromFsToL3(fs, buffer, &HyperRam, &bias_size);
    }
    else
    {
        PRINTF("No weights required for final convolution\n");
    }
}

void layer_free()
{
    pi_ram_free(&HyperRam, (uint32_t)l3_weights, weights_size);
    pi_ram_free(&HyperRam, (uint32_t)l3_bias, bias_size);
}


// The function return L2 memory address where input image should be loader
// Expected format: 128x128xshort
short* layer_init()
{
    L1_Memory = pi_l1_malloc(NULL, _L1_Memory_SIZE);
    if(L1_Memory == NULL)
    {
        PRINTF("WorkingArea alloc error\n");
        return NULL;
    }

    L2_Memory = pi_l2_malloc(_L2_Memory_SIZE);
    if(L2_Memory == NULL)
    {
        PRINTF("L2 Working area alloc error\n");
        return NULL;
    }

    return L2_Memory;
}

short* layer_process(int layer_idx)
{
    PRINTF("layer_process call\n");
    if(layer_idx < NB_CONV)
    {
        //short* weight_base_address;
        //short* weights;
        //short* bias;

        //weight_base_address = layer_output + get_layer_out_size(layer_idx); // expects 3-channels 128x128

        //weights = weight_base_address;
        //bias = weights + weights_size / sizeof(short);

        //PRINTF("Loading weights\n");
        //loadLayerFromL3ToL2(&HyperRam, l3_weights, weights, weights_size);
        //loadLayerFromL3ToL2(&HyperRam, l3_bias, bias, bias_size);
        PRINTF("Convolution\n");
        ConvLayerArray[layer_idx](l2_x, l3_weights, l3_bias, infer_result);
        PRINTF("Convolution done\n");
    }
    else
    {
        PRINTF("Global AvgPool Test\n");
        GPool10(l2_x, infer_result);
    }

    return infer_result;
}

static void cluster_main()
{
    infer_result = layer_process(test_layer_idx);
}

void body(void *parameters)
{
    (void) parameters;

    PRINTF("main call\n");

    struct pi_hyperram_conf hyper_conf;
    pi_hyperram_conf_init(&hyper_conf);
    pi_open_from_conf(&HyperRam, &hyper_conf);

    if (pi_ram_open(&HyperRam))
    {
        PRINTF("Error: cannot open Hyperram!\n");
        pmsis_exit(-2);
    }

    PRINTF("HyperRAM config done\n");

    PRINTF("Configuring Hyperflash and FS..");
    struct pi_device flash;
    struct pi_hyperflash_conf flash_conf;

    pi_hyperflash_conf_init(&flash_conf);
    pi_open_from_conf(&flash, &flash_conf);

    if (pi_flash_open(&flash))
    {
        PRINTF("Error: Flash open failed\n");
        pmsis_exit(-3);
    }

    // The hyper chip needs to wait a bit.
    pi_time_wait_us(100 * 1000);

    struct pi_device fs;
    struct pi_readfs_conf fs_conf;
    pi_readfs_conf_init(&fs_conf);
    fs_conf.fs.flash = &flash;

    pi_open_from_conf(&fs, &fs_conf);

    int error = pi_fs_mount(&fs);
    if (error)
    {
        PRINTF("Error: FS mount failed with error %d\n", error);
        pmsis_exit(-3);
    }

    PRINTF("FS mounted\n");

    int input_size = get_layer_in_size(test_layer_idx) * sizeof(short);
    int output_size = get_layer_out_size(test_layer_idx) * sizeof(short);

    l2_x = pi_l2_malloc(input_size);
    infer_result = pi_l2_malloc(output_size);
    if (l2_x == NULL || infer_result == NULL)
    {
        PRINTF("Error: Failed to allocate %d bytes of L2 memory\n", input_size + output_size);
        pmsis_exit(-4);
    }

    PRINTF("Loading layer %d to HyperRAM\n", test_layer_idx);
    layer_load(&fs, test_layer_idx);

    PRINTF("Unmount FS as it's not needed any more\n");
    pi_fs_unmount(&fs);
    PRINTF("FS unmounted\n");

    char *inputBlob = "../../../input.bin";
    char *outputBlob = "../../../output.bin";

    PRINTF("Init cluster...\n");
    struct pi_device cluster_dev;
    struct pi_cluster_conf cluster_conf;
    struct pi_cluster_task cluster_task;
    pi_cluster_conf_init(&cluster_conf);
    cluster_conf.id = 0;
    cluster_conf.device_type = 0;
    pi_open_from_conf(&cluster_dev, &cluster_conf);
    pi_cluster_open(&cluster_dev);
    PRINTF("Init cluster...done\n");

    //Setting FC to 250MHz
    pi_freq_set(PI_FREQ_DOMAIN_FC, 250000000);

    //Setting Cluster to 175MHz
    // NOTE: Current Gap8 generation does not have clock divider for hyperbus
    // and using FC clocks over 150Mhz is dangerous
    pi_freq_set(PI_FREQ_DOMAIN_CL, 175000000);

    if (layer_init() == NULL)
    {
        pmsis_exit(-5);
    }
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
    pi_fs_file_t* host_file = pi_fs_open(&host_fs, inputBlob, PI_FS_FLAGS_READ);
    if (!host_file)
    {
        PRINTF("Failed to open file, %s\n", inputBlob);
        pmsis_exit(-5);
    }
    PRINTF("Host file open done\n");

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

    PRINTF("Convolution..\n");
#ifdef PERF_COUNT
    unsigned int tm = rt_time_get_us();
#endif
    pi_cluster_send_task_to_cl(&cluster_dev, pi_cluster_task(&cluster_task, (void *)cluster_main, NULL));

#ifdef PERF_COUNT
    tm = rt_time_get_us() - tm;
    PRINTF("Convolution finished in %d us\n", tm);
#else
    PRINTF("Convolution finished\n");
#endif
    int activation_size = get_layer_out_size(test_layer_idx);
    PRINTF("Activations size, shorts: %d\n", activation_size);

    pi_cluster_close(&cluster_dev);

    host_file = pi_fs_open(&host_fs, outputBlob, PI_FS_FLAGS_WRITE);
    if (host_file == NULL)
    {
        PRINTF("Failed to open host file, %s\n", outputBlob);
        pmsis_exit(-7);
    }

    pi_fs_write(host_file, infer_result, activation_size*sizeof(short));
    pi_fs_close(host_file);

    pi_fs_unmount(&host_fs);

    pi_l2_free(l2_x, input_size);
    pi_l2_free(infer_result, output_size);

    layer_free();
}

int main()
{
    PRINTF("Start Single Layer Test\n");
    pmsis_kickoff(body);
    return 0;
}
