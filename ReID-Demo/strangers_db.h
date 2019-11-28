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

#ifndef __STRANGERS_DB_H__
#define __STRANGERS_DB_H__

#include "setup.h"

#define ALLOC_ERROR       1
#define DB_FULL           2
#define DUPLICATE_DROPPED 3

typedef struct Stranger_T
{
    short descriptor[FACE_DESCRIPTOR_SIZE];
    char  name[16];
    char* preview;
} Stranger;

char addStrangerL2(char* preview, short* descriptor);
char addStrangerL3(char* preview, short* descriptor);
char getStranger(int idx, Stranger* s);
void dropStrangers();
char getStrangersCount();

#endif
