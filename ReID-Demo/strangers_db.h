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
