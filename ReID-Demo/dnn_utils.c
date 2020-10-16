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

#include "pmsis.h"
#include "layer_params.h"
#include "dnn_utils.h"

#define IO_BUFF_SIZE 1024

struct pi_device HyperRam;


int loadLayerFromFsToL2(struct pi_device *fs, const char *file_name, void *res, unsigned size)
{
    PRINTF("Loading layer \"%s\" from FS to L2\n", file_name);
    pi_fs_file_t * file = pi_fs_open(fs, file_name, 0);
    if (file == NULL)
    {
        PRINTF("file open failed\n");
        return 0;
    }

    if (file->size > size)
    {
        PRINTF("Provided buffer size %d is smaller than file size %d\n", size, file->size);
        return -1;
    }

    pi_task_t task;
    int size_read = pi_fs_read_async(file, res, file->size, pi_task_block(&task));
    pi_task_wait_on(&task);
    PRINTF("Read %d bytes from %s\n", size_read, file_name);

    pi_fs_close(file);

    return size_read;
}

void* loadLayerFromFsToL3(struct pi_device *fs, const char* file_name, struct pi_device *hyper, int* layer_size)
{
    PRINTF("Loading layer \"%s\" from FS to L3\n", file_name);

    pi_fs_file_t * file = pi_fs_open(fs, file_name, 0);
    if (file == NULL)
    {
        PRINTF("file open failed\n");
        return NULL;
    }
    uint32_t hyper_buff;
    pi_ram_alloc(hyper, &hyper_buff, file->size);
    if(!hyper_buff)
    {
        PRINTF("HyperRAM allocation failed\n");
        return NULL;
    }

    void *buff = pi_l2_malloc(IO_BUFF_SIZE);
    if (buff == NULL)
    {
        return NULL;
    }

    unsigned int size_total = 0;
    unsigned int size = 0;
    pi_task_t task;
    do
    {
        //PRINTF("Reading data to local bufer\n");
        size = pi_fs_read_async(file, buff, IO_BUFF_SIZE, pi_task_block(&task));
        pi_task_wait_on(&task);
        //PRINTF("Read %d bytes from %s\n", size, file_name);
        size = ((size + 3) & ~3);
        if(size)
        {
            //PRINTF("Writing data to L3\n");
            pi_ram_write(hyper, (uint32_t)(hyper_buff+size_total), buff, size);
            // PRINTF("Writing data to L3 done\n");
        }
        size_total += size;
    } while(size_total < file->size);

    pi_l2_free(buff, IO_BUFF_SIZE);

    pi_fs_close(file);

    *layer_size = size_total;

    return (void *)hyper_buff;
}

void loadLayerFromL3ToL2(struct pi_device *hyper, void* hyper_buff, void* base_addr, int layer_size)
{
    pi_cl_ram_req_t req;
    //PRINTF("hyper_buff address: %p\n", hyper_buff);
    //PRINTF("base_addr: %p, size %d\n", base_addr, layer_size);
    pi_cl_ram_read(hyper, (uint32_t)hyper_buff, base_addr, layer_size, &req);
    //PRINTF("after pi_cl_hyper_read\n");
    pi_cl_ram_read_wait(&req);
    //PRINTF("after pi_cl_hyper_read_wait\n");
}

unsigned int l2_distance(const short* v1, const short* v2)
{
    unsigned int sum = 0;

    for (int i = 0; i < FACE_DESCRIPTOR_SIZE; i++)
    {
        int delta = v1[i] - v2[i];
        sum += delta * delta;
    }

    return sum;
}
