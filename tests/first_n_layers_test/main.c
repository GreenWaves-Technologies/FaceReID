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

#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "pmsis.h"
#include "bsp/fs.h"
#include "bsp/fs/readfs.h"
#include "bsp/fs/hostfs.h"
#include "bsp/flash/hyperflash.h"

#if defined(__FREERTOS__)
# include "pmsis_driver_core_api.h"
# include "pmsis_task.h"
# include "pmsis_os.h"
# include "drivers/hyperbus.h"
# include "hyperbus_cl_internal.h"
# include "pmsis_tiling.h"
#else
# include "Gap.h"
#endif

#define IMAGE_WIDTH 128
#define IMAGE_HEIGHT 128

#include "setup.h"

#include "ImgIO.h"
#include "layer_params.h"
#include "network_process.h"
#include "dnn_utils.h"

short* infer_result;
short * l2_x;
short *l2_buffer;

#ifdef PGM_INPUT
L2_MEM unsigned char tmp_buffer[IMAGE_WIDTH*IMAGE_HEIGHT];
#endif

int activation_size = 0;

static void cluster_main()
{
    infer_result = network_process(l2_buffer, &activation_size);
}

void body(void* parameters)
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

    PRINTF("Configuring Hyperflash and FS..\n");
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

    network_load(&fs);

    PRINTF("Unmount FS as it's not needed any more\n");
    pi_fs_unmount(&fs);
    PRINTF("FS unmounted\n");

#ifdef PGM_INPUT
    char *inputBlob = "../../../input.pgm";
#else
    char *inputBlob = "../../../input.bin";
#endif
    char *outputBlob = "../../../output.bin";

    PRINTF("Init cluster...\n");
    struct pi_device cluster_dev;
    struct pi_cluster_conf cluster_conf;
    struct pi_cluster_task cluster_task;

    pi_cluster_conf_init(&cluster_conf);
    cluster_conf.id = 0;
    cluster_conf.device_type = 0;
    pi_open_from_conf(&cluster_dev, &cluster_conf);
    PRINTF("before pi_cluster_open\n");
    pi_cluster_open(&cluster_dev);
    PRINTF("Init cluster...done\n");

    //Setting FC to 250MHz
    pi_freq_set(PI_FREQ_DOMAIN_FC, 250000000);

    //Setting Cluster to 175MHz
    // NOTE: Current Gap8 generation does not have clock divider for hyperbus
    // and using FC clocks over 150Mhz is dangerous
    pi_freq_set(PI_FREQ_DOMAIN_CL, 175000000);

    l2_buffer = pi_l2_malloc(INFERENCE_MEMORY_SIZE);
    if (l2_buffer == NULL)
    {
        PRINTF("Error: Failed to allocate L2 memory\n");
        pmsis_exit(-4);
    }

    l2_x = network_init(&cluster_dev, l2_buffer);
    if (l2_x == NULL)
    {
        pmsis_exit(-1);
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

    pi_fs_file_t* host_file = NULL;
#ifdef PGM_INPUT
    int input_size = IMAGE_WIDTH*IMAGE_HEIGHT;
    unsigned int Wi = IMAGE_WIDTH;
    unsigned int Hi = IMAGE_HEIGHT;
    PRINTF("Reading PGM\n");
    char* tmp_buffer2 = ReadImageFromFile(inputBlob, &Wi, &Hi, tmp_buffer, input_size);
    if(tmp_buffer != tmp_buffer2)
    {
        PRINTF("Failed to read PGM image %dx%d\n", Wi, Hi);
        pmsis_exit(-5);
    }

    for(int i = 0; i < input_size; i++)
    {
        l2_x[i] = tmp_buffer[i];
    }

    PRINTF("Writing input.bin\n");

    host_file = pi_fs_open(&host_fs, "../../../input.bin", PI_FS_FLAGS_WRITE);
    if (host_file == NULL)
    {
        PRINTF("Failed to open file, %s\n", inputBlob);
        pmsis_exit(-6);
    }

    pi_fs_write(host_file, l2_x, input_size*sizeof(short));
    pi_fs_close(host_file);
#else
    host_file = pi_fs_open(&host_fs, inputBlob, PI_FS_FLAGS_READ);
    if (host_file == NULL)
    {
        PRINTF("Failed to open file, %s\n", inputBlob);
        pmsis_exit(-7);
    }
    PRINTF("Host file open done\n");

    int input_size = IMAGE_WIDTH*IMAGE_HEIGHT*sizeof(short);
    int read = pi_fs_read(host_file, l2_x, input_size);
    if(read != input_size)
    {
        PRINTF("Failed to read file %s\n", inputBlob);
        PRINTF("Expected input size %d, but read %d\n", input_size, read);
        pmsis_exit(-8);
    }
    pi_fs_close(host_file);

    PRINTF("Reading input from host...done\n");
#endif

    PRINTF("DNN inference\n");
#ifdef PERF_COUNT
    unsigned int tm = rt_time_get_us();
#endif
    pi_cluster_task(&cluster_task, (void *)cluster_main, NULL);
    cluster_task.slave_stack_size = CL_SLAVE_STACK_SIZE;
    cluster_task.stack_size = CL_STACK_SIZE;
    pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

#ifdef PERF_COUNT
    tm = rt_time_get_us() - tm;
    PRINTF("DNN inference finished in %d us\n", tm);
#endif
    PRINTF("Activations size, shorts: %d\n", activation_size);

    network_deinit(&cluster_dev);
    pi_cluster_close(&cluster_dev);

    host_file = pi_fs_open(&host_fs, outputBlob, PI_FS_FLAGS_WRITE);
    if (host_file == NULL)
    {
        PRINTF("Failed to open file, %s\n", outputBlob);
        pmsis_exit(-9);
    }

    pi_fs_write(host_file, infer_result, activation_size*sizeof(short));
    pi_fs_close(host_file);

    pi_fs_unmount(&host_fs);

    pi_l2_free(l2_buffer, INFERENCE_MEMORY_SIZE);

    network_free();
}

int main()
{
    PRINTF("Start First-n-Layers Test\n");
    pmsis_kickoff(body);
    return 0;
}
