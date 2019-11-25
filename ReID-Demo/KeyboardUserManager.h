#ifndef __KEYBOARD_USER_MANAGER_H__
#define __KEYBOARD_USER_MANAGER_H__

#include "pmsis.h"
#include "setup.h"
#include "strangers_db.h"

int initHandler(struct pi_device* display);
int handleStranger(short* descriptor);

#endif