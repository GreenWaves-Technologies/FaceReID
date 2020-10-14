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
#include <bsp/ram.h>
#include <bsp/ram/hyperram.h>
#include <bsp/flash/hyperflash.h>

#include "dnn_utils.h"
#include "face_db.h"

short face_desk[FACE_DESCRIPTOR_SIZE];
char * person_name;
char out_perf_string[120];

void body(void * parameters)
{
    (void) parameters;

    PRINTF("Start DB Selector test\n");

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

    if (pi_fs_mount(&fs))
    {
        PRINTF("Error: FS mount failed\n");
        pmsis_exit(-3);
    }

    PRINTF("FS mounted\n");

    PRINTF("Loading static ReID database\n");
    char buffer[1024];
    if (!load_static_db(&fs, buffer))
    {
        PRINTF("Static DB load failed!\n");
        pmsis_exit(-4);
    }

    PRINTF("Unmount FS as it's not needed any more\n");
    pi_fs_unmount(&fs);

    char *inputBlob = "../../../input.bin";

    PRINTF("Reading input from host...\n");

    struct pi_hostfs_conf host_fs_conf;
    pi_hostfs_conf_init(&host_fs_conf);
    struct pi_device host_fs;

    pi_open_from_conf(&host_fs, &host_fs_conf);

    if (pi_fs_mount(&host_fs))
    {
        PRINTF("pi_fs_mount failed\n");
        pmsis_exit(-5);
    }

    pi_fs_file_t* host_file = pi_fs_open(&host_fs, inputBlob, PI_FS_FLAGS_READ);
    if (!host_file)
    {
        PRINTF("Failed to open file, %s\n", inputBlob);
        pmsis_exit(-6);
    }
    PRINTF("Host file open done\n");

    int input_size = FACE_DESCRIPTOR_SIZE*sizeof(short);
    int read = pi_fs_read(host_file, face_desk, input_size);
    if(read != input_size)
    {
        PRINTF("Failed to read file %s\n", inputBlob);
        PRINTF("Expected input size %d, but read %d\n", input_size, read);
        pmsis_exit(-7);
    }

    pi_fs_close(host_file);
    pi_fs_unmount(&host_fs);

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
    return pmsis_kickoff(body);
}
