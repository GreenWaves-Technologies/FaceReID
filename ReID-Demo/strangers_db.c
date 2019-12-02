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

#include "strangers_db.h"
#include "dnn_utils.h"

static int global_stranger_idx = 0;

Stranger StrangersDB[STRANGERS_DB_SIZE];

static char findDuplicate(short* descriptor)
{
    char found = 0;
    for(int i = 0; i < global_stranger_idx; i++)
    {
        unsigned int l2 = l2_distance(StrangersDB[i].descriptor, descriptor);
        PRINTF("Comparing with stranger %d, l2 = %u\n", i, l2);
        if(l2 < STRANGER_L2_THRESHOLD)
        {
            found = 1;
            break;
        }
    }

    return found;
}

char addStrangerL2(char* preview, short* descriptor)
{
    uint32_t preview_hyper;
    pi_ram_alloc(&HyperRam, &preview_hyper, 128*128);
    if(!preview_hyper)
    {
        return ALLOC_ERROR;
    }

    int iterations = 128*128 / 1024;
    for(int i = 0; i < iterations; i++)
    {
        pi_ram_write(&HyperRam, preview_hyper+i*1024, preview+i*1024, 1024);
    }

    char status = addStrangerL3((char*)preview_hyper, descriptor);
    if(status != 0)
    {
        pi_ram_free(&HyperRam, preview_hyper, 128*128);
    }

    return status;
}

char addStrangerL3(char* preview, short* descriptor)
{
    if(global_stranger_idx >= 10)
    {
        return DB_FULL;
    }

    if(findDuplicate(descriptor) == 1)
    {
        return DUPLICATE_DROPPED;
    }

    sprintf(StrangersDB[global_stranger_idx].name, "stranger_%d", global_stranger_idx);
    memcpy(StrangersDB[global_stranger_idx].descriptor, descriptor, FACE_DESCRIPTOR_SIZE*sizeof(short));
    StrangersDB[global_stranger_idx].preview = preview;

    PRINTF("Added person %s to strangers list\n", StrangersDB[global_stranger_idx].name);

    global_stranger_idx++;

    return 0;
}

char getStranger(int idx, Stranger* s)
{
    if((idx < 0) || (idx >=10))
    {
        return -1;
    }

    if(s->preview == NULL)
    {
        return -1;
    }

    int iterations = 128*128 / 1024;
    for(int i = 0; i < iterations; i++)
    {
        pi_ram_read(&HyperRam, (uint32_t)(StrangersDB[idx].preview + i*1024), s->preview + i*1024, 1024);
    }

    memcpy(s->descriptor, StrangersDB[idx].descriptor, FACE_DESCRIPTOR_SIZE*sizeof(short));

    memset(s->name, 0, 16);
    strcpy(s->name, StrangersDB[idx].name);

    return 0;
}

void dropStrangers()
{
    for(int i = 0; i < global_stranger_idx; i++)
    {
        if(StrangersDB[i].preview != NULL)
        {
            pi_ram_free(&HyperRam, StrangersDB[i].preview, 128*128);
            StrangersDB[i].preview = NULL;
        }
    }

    global_stranger_idx = 0;
}

char getStrangersCount()
{
    return global_stranger_idx;
}
