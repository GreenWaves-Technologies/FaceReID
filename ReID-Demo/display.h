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

#ifndef DISPLAY_H
#define DISPLAY_H

#include "setup.h"
#include <pmsis.h>

#define LCD_OFF_X 40
#define LCD_OFF_Y 60

#define LCD_TXT_POS_X         2
#define LCD_TXT_POS_Y       220
#define LCD_TXT_HEIGHT(i) ((i) * 8)

#define LCD_TXT_CLR 0x0000 // black
#define LCD_BG_CLR  0xFFFF // white

// Temporary headers for not exported display functions
void writeFillRect(struct pi_device *device, unsigned short x, unsigned short y, unsigned short w, unsigned short h, unsigned short color);
void setCursor(struct pi_device *device, signed short x, signed short y);
void writeText(struct pi_device *device, const char *str, int fontsize);
void setTextColor(struct pi_device *device, uint16_t c);

// Helper functions
int32_t open_display(struct pi_device *display);
void clear_stripe(struct pi_device *display, unsigned posY, unsigned height);
void draw_gwt_logo(struct pi_device *display);
void draw_text(struct pi_device *display, const char *str, unsigned posX, unsigned posY, unsigned fontsize);

#endif
