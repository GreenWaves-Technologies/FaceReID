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