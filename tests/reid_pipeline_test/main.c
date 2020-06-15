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
#include <limits.h>

# include "pmsis.h"

#if defined(__FREERTOS__)
# include "pmsis_driver_core_api.h"
# include "pmsis_task.h"
# include "pmsis_os.h"
# include "drivers/hyperbus.h"
# include "hyperbus_cl_internal.h"
# include "pmsis_tiling.h"
#endif

#include "bsp/bsp.h"
#include "bsp/fs.h"
#include "bsp/fs/readfs.h"
#include "bsp/fs/hostfs.h"
#include "bsp/flash/hyperflash.h"

#include "bsp/gapoc_a.h"

#include "ImgIO.h"

#include "cascade.h"
#include "setup.h"

#include "network_process_manual.h"
#include "dnn_utils.h"
#include "face_db.h"

#include "CnnKernels.h"
#include "ExtraKernels.h"
#include "reid_pipeline.h"

char* tmp_frame_buffer = (char*)(memory_pool+MEMORY_POOL_SIZE) - CAMERA_WIDTH*CAMERA_HEIGHT;
// Largest possible face after Cascade
char* tmp_face_buffer = (char*)(memory_pool+MEMORY_POOL_SIZE) - CAMERA_WIDTH*CAMERA_HEIGHT - 194*194;
char* tmp_img_face_buffer = (char*)(memory_pool+MEMORY_POOL_SIZE) - CAMERA_WIDTH*CAMERA_HEIGHT - 194*194-128*128;

#if defined(CONFIG_GAPOC_A)
char *inputBlob = "../../../input_320x240.pgm";
L2_MEM cascade_reponse_t test_response =
{
    .x = 96,
    .y = 56,
    .w = 128,
    .h = 128,
    .score = 1,
    .layer_idx = 0,
};
#else
char *inputBlob = "../../../input_324x244.pgm";
L2_MEM cascade_reponse_t test_response =
{
    .x = 98,
    .y = 58,
    .w = 128,
    .h = 128,
    .score = 1,
    .layer_idx = 0,
};
#endif

char *outputImage = "../../../output.pgm";
char *outputBlob = "../../../output.bin";

// L2_MEM cascade_reponse_t test_response =
// {
//     .x = 113,
//     .y = 97,
//     .w = 121,
//     .h = 121,
//     .score = 1,
//     .layer_idx = 0,
// };

static void my_copy(short* in, unsigned char* out, int Wout, int Hout)
{
    for(int i = 0; i < Hout; i++)
    {
        for(int j = 0; j < Hout; j++)
        {
            out[i*Wout + j] = (unsigned char)in[i*Wout + j];
        }
    }
}

void body(void * parameters)
{
    (void) parameters;

    PRINTF("Start ReID Pipeline test\n");

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

    PRINTF("Loading layers to HyperRAM\n");
    network_load(&fs);

    PRINTF("Loading static ReID database\n");
    load_static_db(&fs);

    PRINTF("Unmount FS as it's not needed any more\n");
    pi_fs_unmount(&fs);
    PRINTF("Unmount FS done\n");

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

    int input_size = CAMERA_WIDTH*CAMERA_HEIGHT;
    unsigned int Wi = CAMERA_WIDTH;
    unsigned int Hi = CAMERA_HEIGHT;

    PRINTF("Before ReadImageFromFile\n");
    char* read = ReadImageFromFile(inputBlob, &Wi, &Hi, tmp_frame_buffer, input_size);
    PRINTF("After ReadImageFromFile with status: %x\n", read);
    if(read != tmp_frame_buffer)
    {
        PRINTF("Failed\n");
        pmsis_exit(-4);
    }
    PRINTF("Host file read\n");

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

    ArgClusterDnn_T ClusterDnnCall;
    ClusterDnnCall.roi         = &test_response;
    ClusterDnnCall.frame       = tmp_frame_buffer;
    ClusterDnnCall.face        = tmp_face_buffer;
    ClusterDnnCall.scaled_face = network_init(&cluster_dev);
    if(!ClusterDnnCall.scaled_face)
    {
        PRINTF("Failed to initialize ReID network!\n");
        pmsis_exit(-6);
    }

    ExtaKernels_L1_Memory = L1_Memory;

#ifdef PERF_COUNT
    unsigned int tm = rt_time_get_us();
#endif
    PRINTF("Before pi_cluster_send_task_to_cl 1\n");
    pi_cluster_task(&cluster_task, (void (*)(void *))reid_prepare_cluster, &ClusterDnnCall);
    cluster_task.slave_stack_size = CLUSTER_STACK_SIZE;
    cluster_task.stack_size = 2 * CLUSTER_STACK_SIZE;
    pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
    PRINTF("After pi_cluster_send_task_to_cl 1\n");

    my_copy(ClusterDnnCall.scaled_face, tmp_img_face_buffer, 128, 128);

    WriteImageToFile(outputImage, 128, 128, tmp_img_face_buffer);

    PRINTF("Before pi_cluster_send_task_to_cl 2\n");
    pi_cluster_task(&cluster_task, (void (*)(void *))reid_inference_cluster, &ClusterDnnCall);
    cluster_task.slave_stack_size = CLUSTER_STACK_SIZE;
    cluster_task.stack_size = 2 * CLUSTER_STACK_SIZE;
    pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
    PRINTF("After pi_cluster_send_task_to_cl 2\n");

    // Close the cluster
    network_deinit(&cluster_dev);
    pi_cluster_close(&cluster_dev);

    pi_fs_file_t* host_file = pi_fs_open(&host_fs, outputBlob, PI_FS_FLAGS_WRITE);
    if (host_file == NULL)
    {
        PRINTF("Failed to open file, %s\n", outputBlob);
        pmsis_exit(-7);
    }

    pi_fs_write(host_file, ClusterDnnCall.output, ClusterDnnCall.activation_size*sizeof(short));
    pi_fs_close(host_file);

    pi_fs_unmount(&host_fs);

    char* person_name;
    int id_conf = identify_by_db(ClusterDnnCall.output, &person_name);
    printf("Hi, %s! Conf: %d\n", person_name, id_conf);

#ifdef PERF_COUNT
    tm = rt_time_get_us() - tm;
    PRINTF("Cycle time %d microseconds\n", tm);
#endif
}

int main()
{
    PRINTF("Start full ReID pipeline Test\n");
    pmsis_kickoff(body);
    return 0;
}
