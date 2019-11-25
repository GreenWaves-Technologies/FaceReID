#ifndef __BLE_USER_MANAGER_H__
#define __BLE_USER_MANAGER_H__

#include "setup.h"
#include "strangers_db.h"

int initHandler(struct pi_device *gpio_port);
int prepareStranger(void* preveiw);
int handleStranger(short* descriptor);

void admin_body(struct pi_device *display, struct pi_device* gpio_port, uint8_t button_pin);


#endif
