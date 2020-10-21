/*
 * Copyright 2019-2020 GreenWaves Technologies, SAS
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

#ifdef __FREERTOS__
# include "GAPOC_BSP_General.h"
# include "FreeRTOS_util.h"
# include "pmsis_task.h"
#else
# include "bsp/gapoc_a.h"
#endif

#include <fcntl.h>

#include "bsp/fs.h"
#include "bsp/fs/hostfs.h"

#include "setup.h"
#include "display.h"
#include "BleUserManager.h"
#include "strangers_db.h"
#include "face_db.h"
#include "dnn_utils.h"
#include "gaplib/ImgIO.h"

char filename[127];
char initial_name[2][16] = {"Lena", "Francesco"};
char preview[128*128];
short descriptor[512];

unsigned int width = 128, height = 128;

static int open_gpio(struct pi_device *device)
{
    struct pi_gpio_conf gpio_conf;

    pi_gpio_conf_init(&gpio_conf);
    pi_open_from_conf(device, &gpio_conf);

    if (pi_gpio_open(device))
        return -1;

    pi_gpio_pin_configure(device, BUTTON_PIN_ID,    PI_GPIO_INPUT);
    pi_gpio_pin_configure(device, GPIOA2_NINA_RST,  PI_GPIO_OUTPUT);
    pi_gpio_pin_configure(device, GPIOA21_NINA17,   PI_GPIO_OUTPUT);

    pi_gpio_pin_notif_configure(device, BUTTON_PIN_ID, PI_GPIO_NOTIF_FALL);
    pi_gpio_pin_write(device, GPIOA2_NINA_RST,  0);
    pi_gpio_pin_write(device, GPIOA21_NINA17,   1);

    return 0;
}

static void body(void* parameters)
{
    PRINTF("Starting Re-ID body\n");

    pi_pad_set_function(BUTTON_FUNCTION_PIN, 1);

    struct pi_device gpio_port;
    struct pi_gpio_conf gpio_conf;

    pi_gpio_conf_init(&gpio_conf);
    pi_open_from_conf(&gpio_port, &gpio_conf);

    if (open_gpio(&gpio_port))
    {
        PRINTF("Error: cannot open GPIO port\n");
        pmsis_exit(-4);
    }

#if defined(HAVE_DISPLAY)
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

    pi_display_ioctl(&display, PI_ILI_IOCTL_ORIENTATION, (void *)PI_ILI_ORIENTATION_270);
    writeFillRect(&display, 0, 0, 320, 240, 0xFFFF);
    draw_gwt_logo(&display);
#endif

    PRINTF("NINA BLE module test body\n");

    struct pi_hyperram_conf hyper_conf;
    pi_hyperram_conf_init(&hyper_conf);
    pi_open_from_conf(&HyperRam, &hyper_conf);

    if (pi_ram_open(&HyperRam))
    {
        PRINTF("Error: cannot open Hyperram!\n");
        pmsis_exit(-2);
    }

    PRINTF("HyperRAM config done\n");

    printf("Reading input from host...\n");
    struct pi_hostfs_conf host_fs_conf;
    pi_hostfs_conf_init(&host_fs_conf);
    struct pi_device host_fs;

    pi_open_from_conf(&host_fs, &host_fs_conf);

    if (pi_fs_mount(&host_fs))
    {
        PRINTF("pi_fs_mount failed\n");
        pmsis_exit(-4);
    }

    for(int i = 0; i < 2; i++)
    {
        PRINTF("Reading input image...\n");
        sprintf(filename, "../../../%s.pgm", initial_name[i]);
        int res = ReadImageFromFile(filename, width, height, sizeof(char), preview, width*height*sizeof(char), IMGIO_OUTPUT_CHAR, 0);
        if (res != 0)
        {
            PRINTF("Face image load failed\n");
            pmsis_exit(-3);
        }

        PRINTF("Reading input image...done\n");

        sprintf(filename, "../../../%s.bin", initial_name[i]);
        pi_fs_file_t *descriptor_file = pi_fs_open(&host_fs, filename, PI_FS_FLAGS_READ);
        if(!descriptor_file)
        {
            PRINTF("Face descriptor open failed\n");
            pmsis_exit(-5);
        }

        PRINTF("Descriptor opened\n");

        int read = pi_fs_read(descriptor_file, descriptor, FACE_DESCRIPTOR_SIZE * sizeof(short));
        if(read != FACE_DESCRIPTOR_SIZE * sizeof(short))
        {
            PRINTF("Face descriptor read failed\n");
            pmsis_exit(-3);
        }

        PRINTF("Descriptor read\n");

        pi_fs_close(descriptor_file);

        PRINTF("Adding stranger to queue\n");
        addStrangerL2(preview, descriptor);
    }

    unsigned memory_size = (STRANGERS_DB_SIZE + 1) * sizeof(stranger_t) + STRANGERS_DB_SIZE * 128 * 128 + 1024;
    void *ble_buffer = pi_l2_malloc(memory_size);
    if (ble_buffer == NULL)
    {
        PRINTF("Error: Failed to allocate %u bytes of L2\n", memory_size);
        pmsis_exit(-3);
    }

    PRINTF("Waiting for button press event\n");
    while(pi_gpio_pin_notif_get(&gpio_port, BUTTON_PIN_ID) == 0)
    {
        pi_yield();
    }

    PRINTF("Button pressed\n");
    admin_body(&display, &gpio_port, BUTTON_PIN_ID, ble_buffer);

    PRINTF("Dumping Known Faces database for check..\n");
    dump_db();
    PRINTF("Dumping Known Faces database for check..done\n");

    pi_gpio_pin_notif_clear(&gpio_port, BUTTON_PIN_ID);

    pi_l2_free(ble_buffer, memory_size);

    pi_fs_unmount(&host_fs);

    pmsis_exit(0);
}

int main(void)
{
    PRINTF("Start NINA BLE module test\n");
    return pmsis_kickoff((void *) body);
}
