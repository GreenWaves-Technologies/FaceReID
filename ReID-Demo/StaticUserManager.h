#ifndef __STATIC_USER_MANAGER_H__
#define __STATIC_USER_MANAGER_H__

#include "pmsis.h"
#include "strangers_db.h"

int initHandler(struct pi_device* fs);
int prepareStranger(void* preview);
int handleStranger(short* descriptor);


#endif