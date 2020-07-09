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

#include "display.h"
#include "bsp/display/ili9341.h"

int32_t open_display(struct pi_device *display)
{
    struct pi_ili9341_conf ili_conf;

    pi_ili9341_conf_init(&ili_conf);
    pi_open_from_conf(display, &ili_conf);
    if (pi_display_open(display))
        return -1;

    pi_display_ioctl(display, PI_ILI_IOCTL_ORIENTATION, (void *)LCD_ORIENTATION);
    return 0;
}

void clear_stripe(struct pi_device *display, unsigned posY, unsigned height)
{
#if defined(HAVE_DISPLAY)
    writeFillRect(display, 0, posY, LCD_WIDTH, height, LCD_BG_CLR);
#endif
}

void draw_gwt_logo(struct pi_device *display)
{
#if defined(HAVE_DISPLAY)
    setCursor(display, 30, 2);
    writeText(display, "GreenWaves", 3);
    setCursor(display, 10, LCD_TXT_HEIGHT(3) + 2);
    writeText(display, "Technologies", 3);
#endif
}

void draw_text(struct pi_device *display, const char *str, unsigned posX, unsigned posY, unsigned fontsize)
{
#if defined(HAVE_DISPLAY)
    writeFillRect(display, 0, posY, LCD_WIDTH, LCD_TXT_HEIGHT(fontsize), LCD_BG_CLR);
    setCursor(display, posX, posY);
    writeText(display, str, fontsize);
#endif
}
