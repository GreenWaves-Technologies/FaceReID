#ifndef __FACE_DB_H__
#define __FACE_DB_H__

#include "dnn_utils.h"
#include "setup.h"

int load_static_db(struct pi_device * fs);
unsigned int l2_distance(short* v1, short* v2);
int identify_by_db(short* descriptor, char** name);

#ifndef STATIC_FACE_DB
int add_to_db(short* descriptor, char* name);
#endif

int db_free();

void printf_db_descriptors();
void dump_db();

#endif
