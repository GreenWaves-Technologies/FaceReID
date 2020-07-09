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

#include "face_db.h"
#include "dnn_utils.h"

#include <string.h>
#include <fcntl.h>

#include "bsp/fs.h"
#include "bsp/fs/hostfs.h"

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

    PRINTF("Reading face descriptors\n");
    for(int i = 0; i < FACE_DB_SIZE; i++)
    {
        sprintf(buffer, "person_%d.bin", i);
        int status = loadLayerFromFsToL2(fs, buffer, PeopleDescriptors[i], FACE_DESCRIPTOR_SIZE * sizeof(short));
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

int identify_by_db(const short* descriptor, char** name)
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

static int find_in_db(const short *descriptor)
{
    int i;
    for (i = 0; i < identified_people; i++)
    {
        if (memcmp(descriptor, PeopleDescriptors[i], FACE_DESCRIPTOR_SIZE * sizeof(short)) == 0)
        {
            PRINTF("Found descriptor with index: %d (%s)\n", i, PeopleNames[i]);
            return i;
        }
    }

    return -1;
}

int add_to_db(const short* descriptor, const char* name)
{
    int i = find_in_db(descriptor);
    if (i >= 0) // Only update name if the descriptor is already set
    {
        int j;
        for (j = 0; (j < 16) && (name[j] != '\0'); j++)
            PeopleNames[i][j] = name[j];
        if (j == 16)
            j--;
        PeopleNames[i][j] = '\0';

        return i;
    }

    if (identified_people >= FACE_DB_SIZE)
    {
        PRINTF("Error: Descriptor DB is full\n");
        return -1;
    }

    memcpy(PeopleDescriptors[identified_people], descriptor, FACE_DESCRIPTOR_SIZE * sizeof(short));
    int j;
    for (j = 0; (j < 16) && (name[j] != '\0'); j++)
        PeopleNames[identified_people][j] = name[j];
    if (j == 16)
        j--;
    PeopleNames[identified_people][j] = '\0';

    return identified_people++;
}

int drop_from_db(const short * descriptor)
{
    int i = find_in_db(descriptor);
    if (i < 0)
    {
        return -1;
    }

    identified_people--;
    if (i != identified_people)
    {
        memcpy(PeopleDescriptors[i], PeopleDescriptors[identified_people], FACE_DESCRIPTOR_SIZE * sizeof(short));
        memcpy(PeopleNames[i], PeopleNames[identified_people], 16 * sizeof(char));
    }

    return i;
}

char get_identities_count(void)
{
    return identified_people;
}

char get_identity(int idx, short ** descriptor, char ** name)
{
    if ((idx < 0) || (idx >= identified_people))
    {
        return -1;
    }

    if (name != NULL)
        *name = PeopleNames[idx];
    if (descriptor != NULL)
        *descriptor = PeopleDescriptors[idx];

    return 0;
}

void printf_db_descriptors(void)
{
    for(int i = 0; i < FACE_DB_SIZE; i++)
    {
        PRINTF("descriptor %d: %d, %d, %d\n", i, PeopleDescriptors[i][0], PeopleDescriptors[i][1], PeopleDescriptors[i][2]);
    }
}

void dump_db(void)
{
    struct pi_hostfs_conf host_fs_conf;
    pi_hostfs_conf_init(&host_fs_conf);
    struct pi_device host_fs;

    pi_open_from_conf(&host_fs, &host_fs_conf);

    if (pi_fs_mount(&host_fs))
    {
        PRINTF("pi_fs_mount failed\n");
        pmsis_exit(-4);
    }

    char string_buffer[64];
    for(int i = 0; i < identified_people; i++)
    {
        sprintf(string_buffer, "../../../%s.bin", PeopleNames[i]);

        PRINTF("Writing descriptor file \"%s\" ...", string_buffer);

        pi_fs_file_t* descriptor_file = pi_fs_open(&host_fs, string_buffer, PI_FS_FLAGS_WRITE);
        if(!descriptor_file)
        {
            PRINTF("open failed\n");
            pmsis_exit(-100);
        }

        int byte_written = pi_fs_write(descriptor_file, PeopleDescriptors[i], FACE_DESCRIPTOR_SIZE * sizeof(short));
        if (byte_written != FACE_DESCRIPTOR_SIZE * sizeof(short))
        {
            PRINTF("write failed\n");
            pmsis_exit(-100);
        }

        pi_fs_close(descriptor_file);
        PRINTF("done\n");
    }

    pi_fs_unmount(&host_fs);
}
