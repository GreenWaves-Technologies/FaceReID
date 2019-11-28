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
#include "bsp/flash/hyperflash.h"

#if defined(__FREERTOS__)
# include "pmsis_l2_malloc.h"
# include "pmsis_driver_core_api.h"
# include "pmsis_task.h"
# include "pmsis_os.h"
# include "drivers/hyperbus.h"
# include "hyperbus_cl_internal.h"
#else
# include "extra_emul_stubs.h"
#endif

#include "dnn_utils.h"
#include "face_db.h"

short face_desk[FACE_DESCRIPTOR_SIZE];
char * person_name;
char out_perf_string[120];

void body(void * parameters)
{
    (void) parameters;
    int File = 0;
    struct pi_device cluster_dev;
    struct pi_cluster_conf cluster_conf;
    struct pi_cluster_task cluster_task;
    struct pi_hyper_conf hyper_conf;

    PRINTF("Start DB Selector test\n");

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

    PRINTF("Configuring Hyperflash and FS..\n");
    struct pi_device fs;
    struct pi_device flash;
    struct pi_fs_conf conf;
    struct pi_hyperflash_conf flash_conf;
    pi_fs_conf_init(&conf);

    pi_hyperflash_conf_init(&flash_conf);
    pi_open_from_conf(&flash, &flash_conf);

    if (pi_flash_open(&flash))
    {
        PRINTF("Error: Flash open failed\n");
        pmsis_exit(-3);
    }
    conf.flash = &flash;

    pi_open_from_conf(&fs, &conf);

    if (pi_fs_mount(&fs))
    {
        PRINTF("Error: FS mount failed\n");
        pmsis_exit(-3);
    }

    PRINTF("FS mounted\n");

    PRINTF("Loading static ReID database\n");
    if (!load_static_db(&fs))
    {
        PRINTF("Static DB load failed!\n");
        pmsis_exit(-4);
    }

    PRINTF("Unmount FS as it's not needed any more\n");
    pi_fs_unmount(&fs);

    char *inputBlob = "../../../input.bin";

    rt_bridge_connect(1, NULL);

    File = rt_bridge_open(inputBlob, 0, 0, NULL);
    if (File == 0)
    {
        PRINTF("Failed to open file, %s\n", inputBlob);
        pmsis_exit(-6);
    }
    int input_size = FACE_DESCRIPTOR_SIZE*sizeof(short);
    int read = rt_bridge_read(File, face_desk, input_size, NULL);
    if(read != input_size)
    {
        PRINTF("Failed to read file %s\n", inputBlob);
        PRINTF("Expected input size %d, but read %d\n", input_size, read);
        pmsis_exit(-7);
    }

    rt_bridge_close(File, NULL);
    rt_bridge_disconnect(NULL);

    int id_l2 = identify_by_db(face_desk, &person_name);

    sprintf(out_perf_string, "Hi, %s!\n", person_name);
    PRINTF(out_perf_string);
    sprintf(out_perf_string, "ReID L2: %d\n", id_l2);
    PRINTF(out_perf_string);

    pmsis_exit(0);
}

int main()
{
    PRINTF("Start Single Layer Test\n");
    pmsis_kickoff(body);
    return 0;
}
