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

#include "setup.h"

#include "bsp/bsp.h"
#include "bsp/buffer.h"
#include "bsp/display/ili9341.h"
#include "pmsis.h"

#include "StaticUserManagerBleNotifier.h"
#include "face_db.h"

#define STRANGER_MESSAGE 0x10
#define USER_MESSAGE 0x20

uint8_t message[18];
pi_nina_ble_t ble;

int initHandler(struct pi_device* fs, struct pi_device* display)
{
    PRINTF("Loading static ReID database\n");
    int status = load_static_db(fs);
    if(!status)
    {
        PRINTF("Static DB load failed!\n");
        return status;
    }

    PRINTF("Enabling BLE module...\n");

    pi_pad_set_function(CONFIG_HYPERBUS_DATA6_PAD, CONFIG_UART_RX_PAD_FUNC);

#if defined(HAVE_DISPLAY)
    setCursor(display, 0, 220);
    writeFillRect(display, 0, LCD_OFF_Y, 240, 320, 0xFFFF);
    writeText(display, "Enabling BLE", 2);
#endif

    uint8_t rx_buffer[PI_AT_RESP_ARRAY_LENGTH];

    #if defined(__FREERTOS__)
    // NOTICE:
    // With current silicon there may be problems to use UART Rx (GAP8 receiving) while HyperBus interface is
    // enabled. To use UART Rx, pin B7 (=HYPER_DQ[6] when used as HyperBus I/O) must be configured in its default functionality (Alt.0),
    // which means HyperBus is not usable at this time.
    // In other words, usage of HyperBus and UART Rx must be time multiplexed, toggling the functionality of pin B7.

    // To limit power consumption from HyperMem without initializing HyperBus interface,
    // you may pull its nCS low (inactive) by using GPIO mode, e.g. as follows:
    //GAPOC_GPIO_Init_Pure_Output_High(GPIO_A30);  // CSN0 = GPIO30 on B15
    //GAPOC_GPIO_Init_Pure_Output_High(GPIO_A31);  // CSN1 = GPIO31 on A16

    // Bug work-around (see above):
    // Set pin B7 for default behavior (HyperBus not usable then)
    GAPOC_AnyPin_Config( B7, NOPULL, uPORT_MuxAlt0 );  // pin GAP_B7 keeps default function = SPIM0_SCK (output)
    #endif

    pi_nina_b112_open(&ble);

    PRINTF("BLE UART init done\n");

    #ifdef __FREERTOS__
    // Init GPIO that will control NINA DSR in deasserted position
    GAPOC_GPIO_Init_Pure_Output_Low(GAPOC_NINA17_DSR);

    // Enable BLE (release reset)
    GAPOC_GPIO_Set_High(GAPOC_NINA_NRST);

    vTaskDelay( 1 * 1000 / portTICK_PERIOD_MS ); // Delay in ms(1 tick/ms).
    // Now release GPIO_LED_G/NINA_SW1 so it can be driven by NINA
    GAPOC_GPIO_Init_HighZ(GPIO_A1_B2);
    #else
    // Init GPIO that will control NINA DSR in deasserted position
    rt_gpio_set_pin_value(0, GPIOA21_NINA17, 0);

    // Enable BLE (release reset)
    rt_gpio_set_pin_value(0, GPIOA2_NINA_RST, 1);

    pi_time_wait_us(1*1000*1000);
    #endif

    // Initiliaze NINA as BLE Peripheral

    PRINTF("Sending cmd using pmsis bsp\n");
    pi_nina_b112_AT_send(&ble, "E0");
    PRINTF("Echo disabled\n");
    pi_nina_b112_AT_send(&ble, "+UFACTORY");
    PRINTF("Factory configuration restored\n");
    pi_nina_b112_AT_send(&ble, "+UBTUB=FFFFFFFFFFFF");
    PRINTF("Set UBTUB\n");
    pi_nina_b112_AT_send(&ble, "+UBTLE=2");
    PRINTF("Set UBTLE\n");
    pi_nina_b112_AT_send(&ble, "+UBTLN=GreenWaves-GAPOC");
    PRINTF("Set UBTLN\n");
    pi_nina_b112_AT_query(&ble, "+UMRS?", (char *) rx_buffer);
    PRINTF("BLE configuration : %s\n", rx_buffer);
    pi_nina_b112_AT_query(&ble, "+UBTLN?", (char *) rx_buffer);
    PRINTF("BLE name : %s\n", rx_buffer);
    //pi_nina_b112_close(&ble);

    PRINTF("AT Config Done\n");

#if defined(HAVE_DISPLAY)
    setCursor(display, 0, 220);
    writeFillRect(display, 0, 220, 240, 8*2, 0xFFFF);
    writeText(display, "Waiting for client", 2);
#endif

    pi_nina_b112_wait_for_event(&ble, rx_buffer);
    PRINTF("Received Event after reboot: %s\n", rx_buffer);

    // Enter Data Mode
    pi_nina_b112_AT_send(&ble, "O");
    PRINTF("Data Mode Entered!\n");

#if defined(HAVE_DISPLAY)
    setCursor(display, 0, 220);
    writeFillRect(display, 0, 220, 240, 8*2, 0xFFFF);
    writeText(display, "Client connected", 2);
#endif

    pi_pad_set_function(CONFIG_HYPERBUS_DATA6_PAD, CONFIG_HYPERRAM_DATA6_PAD_FUNC);

    return status;
}

int handleStranger(short* descriptor)
{
    (void) descriptor;

    pi_pad_set_function(CONFIG_HYPERBUS_DATA6_PAD, CONFIG_UART_RX_PAD_FUNC);

    message[0] = STRANGER_MESSAGE;
    pi_nina_b112_send_data_blocking(&ble, message, 1);

    pi_pad_set_function(CONFIG_HYPERBUS_DATA6_PAD, CONFIG_HYPERRAM_DATA6_PAD_FUNC);

    return 0;
}

int handleUser(char* name)
{
    pi_pad_set_function(CONFIG_HYPERBUS_DATA6_PAD, CONFIG_UART_RX_PAD_FUNC);

    message[0] = USER_MESSAGE;
    for(int i = 0; i < 16; i++)
        message[i+1] = name[i];

    pi_nina_b112_send_data_blocking(&ble, message, 1+16);
    // waiting for sound playback to continue
    pi_nina_b112_get_data_blocking(&ble, message, 1);

    pi_pad_set_function(CONFIG_HYPERBUS_DATA6_PAD, CONFIG_HYPERRAM_DATA6_PAD_FUNC);

    return 0;
}

void closeHandler()
{
    pi_pad_set_function(CONFIG_HYPERBUS_DATA6_PAD, CONFIG_UART_RX_PAD_FUNC);

    pi_nina_b112_close(&ble);

    pi_pad_set_function(CONFIG_HYPERBUS_DATA6_PAD, CONFIG_HYPERRAM_DATA6_PAD_FUNC);
}
