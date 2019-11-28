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

#include "Gap.h"
#include "setup.h"
#include "ui_input_box.h"
#include "PS2Keyboard.h"

unsigned char textbox_ui_buffer[INPUT_BOX_MAX_LENGTH+1] = {0};
unsigned short textbox_usage = 0;

int is_printable(unsigned char c)
{
    int is_digit = ((c >= '0') && (c <= '9'));
    int is_letter = (((c >= 'a') && (c <= 'z')) || ((c >= 'A') && (c <= 'Z')));
    return is_digit || is_letter;
}

void textbox_add_symbol(unsigned char c)
{
    if(textbox_usage < INPUT_BOX_MAX_LENGTH)
    {
        textbox_ui_buffer[textbox_usage] = c;
        textbox_usage++;
        textbox_ui_buffer[textbox_usage] = '\0';
    }
    else
    {
        for(int i = 0; i < INPUT_BOX_MAX_LENGTH-1; i++)
        {
            textbox_ui_buffer[i] = textbox_ui_buffer[i+1];
        }
        textbox_ui_buffer[INPUT_BOX_MAX_LENGTH-1] = c;
    }
}

void textbox_backspace()
{
    if(textbox_usage == 0)
    {
        return;
    }

    textbox_usage--;
    textbox_ui_buffer[textbox_usage] = '\0';
}

char* input_box(struct pi_device* display, int top, char* caption)
{
    textbox_usage = 0;
    textbox_ui_buffer[0] = '\0';
    writeFillRect(display, 0, top, 240, 8*2, 0xFFFF);
    writeFillRect(display, 0, top+20, 240, 8*2, 0xFFFF);

    setCursor(display, 0, top);
    writeText(display, caption, 2);

    while(1)
    {
        if(kb_available())
        {
            unsigned char key = kb_read();
            PRINTF("Pressed button with code: %d\n", key);
            if(is_printable(key))
            {
                textbox_add_symbol(key);
            }
            else if (key == PS2_BACKSPACE)
            {
                textbox_backspace();
                writeFillRect(display, 0, top+20, 240, 8*2, 0xFFFF); // 2 - letter size mutiplier in writeText
            }
            else if (key == PS2_ENTER)
            {
                PRINTF("Enter pressed! Processing!\n");
                return (char*)textbox_ui_buffer;
            }
            else if (key == PS2_ESC)
            {
                PRINTF("ESC pressed! Canceling!\n");
                return NULL;
            }
        }

        setCursor(display, 0, 220);
        writeText(display, (char*)textbox_ui_buffer, 2);
    }

    return NULL;
}
