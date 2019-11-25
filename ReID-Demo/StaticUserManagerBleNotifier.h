#ifndef __STATIC_USER_MANAGER_BLE_NOTIFIER_H__
#define __STATIC_USER_MANAGER_BLE_NOTIFIER_H__

#include "pmsis.h"
#include "strangers_db.h"

int initHandler(struct pi_device* fs, struct pi_device* display);
int handleStranger(short* descriptor);
int handleUser(char* name);
void closeHandler();

#endif
