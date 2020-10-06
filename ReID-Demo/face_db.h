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

#ifndef __FACE_DB_H__
#define __FACE_DB_H__

#include "bsp/fs.h"
#include "setup.h"

int load_static_db(struct pi_device *fs, char *buffer);
int identify_by_db(const short *descriptor, char** name);

#ifndef STATIC_FACE_DB
int add_to_db(const short *descriptor, const char *name);
int drop_from_db(const short * descriptor);
#endif

char get_identities_count(void);
char get_identity(int idx, short ** descriptor, char ** name);

void printf_db_descriptors(void);
void dump_db(void);

#endif
