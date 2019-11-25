#include <stdlib.h>
#include "setup.h"
#include "KeyboardUserManager.h"
#include "PS2Keyboard.h"
#include "ui_input_box.h"

struct pi_device* display;
int initHandler(struct pi_device* _display)
{
    display = _display;
    kb_begin(3, 2);
    return !0;
}

int handleStranger(short* descriptor)
{
    char string_buffer[64];
    writeText(display, "STOP and introduce!", 2);
    PRINTF("STOP and introduce!\n");

    PRINTF("Waiting for any key..\n");
    unsigned int now = rt_time_get_us();
    // 5 seconds timeout to make a decision
    while(abs(rt_time_get_us()-now) < 5*1000000)
    {
        if(kb_available())
        {
            unsigned char key = kb_read();
            if(key == PS2_ENTER)
            {
                char* name = input_box(display, 200, "Name (<=16 letters):");
                if(name)
                {
                    int db_status = add_to_db(descriptor, name);
                    writeFillRect(display, 0, 200, 240, 8*2, 0xFFFF);
                    writeFillRect(display, 0, 220, 240, 8*2, 0xFFFF);
                    setCursor(display, 0, 200);
                    if(db_status >= 0)
                    {
                        sprintf(string_buffer, "Person ID: %d", db_status);
                        return 0;
                    }
                    else
                    {
                        sprintf(string_buffer, "Err %d, DB Overflow!", db_status);
                        return DB_FULL;
                    }
                    writeText(display, string_buffer, 2);
                    PRINTF(string_buffer);

                    break;
                }
            }
            else if(key == PS2_ESC)
            {
                writeFillRect(display, 0, 200, 240, 8*2, 0xFFFF);
                writeFillRect(display, 0, 220, 240, 8*2, 0xFFFF);
                return DUPLICATE_DROPPED;
            }
        }
    }
}