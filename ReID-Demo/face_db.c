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

#include "face_db.h"
#include "dnn_utils.h"

#include <fcntl.h>

#ifdef STATIC_FACE_DB
int identified_people = FACE_DB_SIZE;
#else
int identified_people = 0;
#endif

short PeopleDescriptors[FACE_DB_SIZE][FACE_DESCRIPTOR_SIZE];
char PeopleNames[FACE_DB_SIZE][16];

int load_static_db(struct pi_device * fs)
{
    char buffer[64];
    int descriptors = 0;
    int names = 0;

    PRINTF("Reading face descriptors\n");
    for(int i = 0; i < FACE_DB_SIZE; i++)
    {
        sprintf(buffer, "person_%d.bin", i);
        int status = loadLayerFromFsToL2(fs, buffer, PeopleDescriptors[i], FACE_DESCRIPTOR_SIZE*sizeof(short));
        if(status <= 0)
        {
            PRINTF("Person %d descriptor read failed with status %d\n", i, status);
            return 0;
        }
        descriptors++;
    }

    char* names_buffer = (char*)memory_pool;

    PRINTF("Reading names index\n");
    int names_buffer_size = loadLayerFromFsToL2(fs, "index.txt", names_buffer, 16*FACE_DB_SIZE);
    if(names_buffer_size <= 0)
    {
        PRINTF("names index read failed\n");
        return 0;
    }

    int name_idx = 0;
    int current_name_symbol = 0;

    for(int i = 0; i < names_buffer_size; i++)
    {
        if(names_buffer[i] != '\n')
        {
            PeopleNames[name_idx][current_name_symbol] = names_buffer[i];
            current_name_symbol++;
        }
        else
        {
            PeopleNames[name_idx][current_name_symbol] = '\0';
            name_idx++;
            current_name_symbol = 0;
        }

        if(name_idx >= FACE_DB_SIZE)
        {
            break;
        }
    }

    PRINTF("Loaded descriptors for:\n");
    for(int i = 0; i < FACE_DB_SIZE; i++)
    {
        PRINTF("\t%s\n", PeopleNames[i]);
    }

    return descriptors;
}

int identify_by_db(short* descriptor, char** name)
{
    if(identified_people == 0)
    {
        return -1;
    }

    unsigned int min_l2 = -1;
    int min_l2_idx = -1;

    for(int i = 0; i < identified_people; i++)
    {
        unsigned int l2 = l2_distance(PeopleDescriptors[i], descriptor);
        PRINTF("L2 distance for %d (%s): %u\n", i, PeopleNames[i], l2);
        if(l2 < min_l2)
        {
            min_l2 = l2;
            min_l2_idx = i;
        }
    }

    PRINTF("Found person with index %d from %d people\n", min_l2_idx, identified_people);

    *name = PeopleNames[min_l2_idx];
    return min_l2;
}

int add_to_db(short* descriptor, char* name)
{
    if(identified_people < FACE_DB_SIZE)
    {
        for(int i = 0; i < FACE_DESCRIPTOR_SIZE; i++)
        {
            PeopleDescriptors[identified_people][i] = descriptor[i];
        }
        int i = 0;
        while((i < 16) && (name[i] !='\0'))
        {
            PeopleNames[identified_people][i] = name[i];
            i++;
        }

        if(i == 16) i--;

        PeopleNames[identified_people][i] = '\0';

        identified_people++;

        return identified_people-1;
    }
    else
    {
        return -1;
    }
}

void printf_db_descriptors()
{
    for(int i = 0; i < FACE_DB_SIZE; i++)
    {
        PRINTF("descriptor %d: %d, %d, %d\n", i, PeopleDescriptors[i][0], PeopleDescriptors[i][1], PeopleDescriptors[i][2]);
    }
}

void dump_db()
{
    char string_buffer[64];
    for(int i = 0; i < identified_people; i++)
    {
        sprintf(string_buffer, "../../../%s.bin", PeopleNames[i]);
        PRINTF("Writing descriptor file \"%s\" ..\n", string_buffer);

        int descriptor_file = rt_bridge_open(string_buffer, O_RDWR | O_CREAT, S_IRWXU, NULL);
        if(descriptor_file < 0)
        {
            PRINTF("Face descriptor open failed\n");
            pmsis_exit(0);
        }

        int bridge_status = rt_bridge_write(descriptor_file, PeopleDescriptors[i], 512*sizeof(short), NULL);

        if(bridge_status != 512*sizeof(short))
        {
            PRINTF("Face descriptor read failed\n");
            pmsis_exit(0);
        }

        rt_bridge_close(descriptor_file, NULL);
        PRINTF("Writing descriptor file..done\n");
    }
}
