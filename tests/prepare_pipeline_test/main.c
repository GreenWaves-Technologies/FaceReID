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
#include <limits.h>

#include "pmsis.h"

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

#include "setup.h"
#include "cascade.h"

#include "network_process.h"
#include "dnn_utils.h"
#include "reid_pipeline.h"
#include "ImgIO.h"

char* tmp_frame_buffer = (char*)(memory_pool+MEMORY_POOL_SIZE) - CAMERA_WIDTH*CAMERA_HEIGHT;
// Largest possible face after Cascade
char* tmp_face_buffer = (char*)(memory_pool+MEMORY_POOL_SIZE) - CAMERA_WIDTH*CAMERA_HEIGHT - 194*194;
char* tmp_img_face_buffer = (char*)(memory_pool+MEMORY_POOL_SIZE) - CAMERA_WIDTH*CAMERA_HEIGHT - 194*194-128*128;

#if defined(CONFIG_GAPOC_A)
char *inputBlob = "../../../input_320x240.pgm";
L2_MEM cascade_response_t test_response_l0 =
{
    .x = 124,
    .y = 80,
    .w = 120,
    .h = 120,
    .score = 1,
    .layer_idx = 0,
};

L2_MEM cascade_response_t test_response_l1 =
{
    .x = 120,
    .y = 48,
    .w = 152,
    .h = 152,
    .score = 1,
    .layer_idx = 1,
};

L2_MEM cascade_response_t test_response_l2 =
{
    .x = 98,
    .y = 18,
    .w = 194,
    .h = 194,
    .score = 1,
    .layer_idx = 2,
};

#else
char *inputBlob = "../../../input_324x244.pgm";
L2_MEM cascade_response_t test_response_l0 =
{
    .x = 126,
    .y = 82,
    .w = 120,
    .h = 120,
    .score = 1,
    .layer_idx = 0,
};

L2_MEM cascade_response_t test_response_l1 =
{
    .x = 122,
    .y = 50,
    .w = 152,
    .h = 152,
    .score = 1,
    .layer_idx = 1,
};

L2_MEM cascade_response_t test_response_l2 =
{
    .x = 100,
    .y = 20,
    .w = 194,
    .h = 194,
    .score = 1,
    .layer_idx = 2,
};

#endif
char *outputBlob = "../../../output.pgm";

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

void body(void* parameters)
{
    (void) parameters;

    PRINTF("Start Prepare Pipeline test\n");

    struct pi_hyperram_conf hyper_conf;
    pi_hyperram_conf_init(&hyper_conf);
    pi_open_from_conf(&HyperRam, &hyper_conf);

    if (pi_ram_open(&HyperRam))
    {
        PRINTF("Error: cannot open Hyperram!\n");
        pmsis_exit(-2);
    }

    PRINTF("HyperRAM config done\n");

    PRINTF("Reading image from host...\n");

    int input_size = CAMERA_WIDTH*CAMERA_HEIGHT;
    unsigned int Wi = CAMERA_WIDTH;
    unsigned int Hi = CAMERA_HEIGHT;
    PRINTF("Before ReadImageFromFile\n");
    unsigned char* read = ReadImageFromFile(inputBlob, &Wi, &Hi, tmp_frame_buffer, input_size);
    PRINTF("After ReadImageFromFile with status: 0x%p\n", read);
    if(read != tmp_frame_buffer)
    {
        PRINTF("Image read failed\n");
        pmsis_exit(-3);
    }
    PRINTF("Reading image from host...done\n");

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

    ArgClusterDnn_T ClusterDnnCall;
    ClusterDnnCall.roi         = &TEST_RESPONSE;
    ClusterDnnCall.frame       = tmp_frame_buffer;
    ClusterDnnCall.face        = tmp_face_buffer;
    ClusterDnnCall.scaled_face = network_init(&cluster_dev);
    if(!ClusterDnnCall.scaled_face)
    {
        PRINTF("Failed to initialize ReID network!\n");
        pmsis_exit(-5);
    }

    pi_cluster_task(&cluster_task, (void *)reid_prepare_cluster, &ClusterDnnCall);
    cluster_task.slave_stack_size = CL_SLAVE_STACK_SIZE;
    cluster_task.stack_size = CL_STACK_SIZE;
    pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

    network_deinit(&cluster_dev);
    pi_cluster_close(&cluster_dev);

    my_copy(ClusterDnnCall.scaled_face, tmp_img_face_buffer, 128, 128);

    PRINTF("Writing output to file\n");
    WriteImageToFile(outputBlob, 128, 128, tmp_img_face_buffer);
    WriteImageToFile("../../../tmp.pgm", ClusterDnnCall.roi->w, ClusterDnnCall.roi->h, ClusterDnnCall.face);
    PRINTF("Writing output to file..done\n");

    pmsis_exit(0);
}

int main()
{
    PRINTF("Start Prepare Pipeline Test\n");
    pmsis_kickoff(body);
    return 0;
}
