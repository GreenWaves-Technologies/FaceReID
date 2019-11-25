#include "pmsis.h"
#include "bsp/display/ili9341.h"
#include "setup.h"
#include "PS2Keyboard.h"
#include "ui_input_box.h"

int main()
{
    PRINTF("Keyboard test main\n");

    PRINTF("Initializing display\n");
    struct pi_ili9341_conf ili_conf;
    pi_ili9341_conf_init(&ili_conf);
    struct pi_device display;
    pi_open_from_conf(&display, &ili_conf);
    if (pi_display_open(&display))
    {
        PRINTF("Error: display init failed\n");
        pmsis_exit(-5);
    }
    PRINTF("Initializing display done\n");

    pi_display_ioctl(&display, ILI_IOCTL_ORIENTATION, ILI_ORIENTATION_0);
    writeFillRect(&display, 0, 0, 240, 320, 0xFFFF);

    PRINTF("before kb_begin\n");
    kb_begin(3, 2);
    PRINTF("After kb_begin\n");

    while(1)
    {
        char* message = input_box(&display, 200, "Please, input name:");
        if(message)
        {
            PRINTF("%s\n", message);
        }
        else
        {
            PRINTF("Input aborted!\n");
        }
    }

    return 0;
}