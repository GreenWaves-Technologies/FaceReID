#include "pmsis.h"
#include "setup.h"
#include "PS2Keyboard.h"

int main()
{
    PRINTF("Keyboard test main\n");

    if (rt_event_alloc(NULL, 1))
        return -1;

    rt_padframe_profile_t *profile_gpio = rt_pad_profile_get("hyper_gpio");

    if (profile_gpio == NULL)
    {
        PRINTF("pad config error\n");
        return 1;
    }

    rt_padframe_set(profile_gpio);

    kb_begin(3, 2);

    while(1)
    {
        if(kb_available())
        {
            int dat = kb_read();
            PRINTF("Pressed button with code: %d\n", dat);
        }
    }

    return 0;
}