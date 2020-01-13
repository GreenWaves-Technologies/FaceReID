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

#include "bsp/gapoc_a.h"
#include "bsp/ram/hyperram.h"
#include "bsp/ble/nina_b112/nina_b112.h"

#include "setup.h"
#include "BleUserManager.h"
#include "ble_protocol.h"
#include "strangers_db.h"
#include "face_db.h"
#include "dnn_utils.h"
#include "display.h"

uint8_t empty_response = '\0';
uint8_t ack = BLE_ACK;
uint8_t action = 0;
volatile uint8_t ble_exit = 0;
rt_timer_t ble_timer;

typedef struct BleContext_T
{
    int queue_head;
    int queue_tail;
    char read_mode;
    char write_mode;
    int face_chunk_idx;
    struct pi_device* display;
    pi_nina_ble_t* ble;
    Stranger* l2_strangers;
    char* current_name;
    short* current_descriptor;

} BleContext;

#define MIN(a, b) (((a)<(b))?(a):(b))

void ble_protocol_handler(void* params)
{
    BleContext* context = (BleContext*)params;

    rt_timer_stop(&ble_timer);

    switch(action)
    {
        case BLE_READ:
            PRINTF("BLE_READ request got\n");
            PRINTF("Queue head: %d, Queue tail: %d\n", context->queue_head, context->queue_tail);

            if(context->queue_head != context->queue_tail)
            {
#if defined(HAVE_DISPLAY)
                char message[32];
                sprintf(message, "Sending %d/%d", context->queue_head+1, context->queue_tail);
                draw_text(context->display, message, LCD_TXT_POS_X, LCD_TXT_POS_Y, 2);
#endif

                // there is something in queue
                context->read_mode = 1;
                pi_nina_b112_send_data_blocking(context->ble, &ack, 1);
                PRINTF("BLE_ACK responded\n");
            }
            else
            {
                PRINTF("Nothing to read\n");
                draw_text(context->display, "Ready", LCD_TXT_POS_X, LCD_TXT_POS_Y, 2);
                pi_nina_b112_send_data_blocking(context->ble, &empty_response, 1);
            }
            break;
        case BLE_GET_NAME:
            PRINTF("BLE_GET_NAME request got\n");
            if(context->read_mode && (context->queue_head != context->queue_tail)) // we are reading and have something in queue
            {
                pi_nina_b112_send_data_blocking(context->ble, (uint8_t *) context->l2_strangers[context->queue_head].name, 16);
                PRINTF("Name %s responded\n", context->l2_strangers[context->queue_head].name);
            }
            else
            {
                PRINTF("ERROR: Empty respond sent\n");
                pi_nina_b112_send_data_blocking(context->ble, &empty_response, 1);
            }
            break;
        case BLE_GET_PHOTO:
            PRINTF("BLE_GET_PHOTO request got\n");
            if(context->read_mode && (context->queue_head != context->queue_tail)) // we are reading and have something in queue
            {
                char* ptr = (char *) (context->l2_strangers[context->queue_head].preview + context->face_chunk_idx * DATA_CHUNK_SIZE);
                int size = MIN(DATA_CHUNK_SIZE, 128 * 128 - context->face_chunk_idx * DATA_CHUNK_SIZE);
                pi_nina_b112_send_data_blocking(context->ble,(uint8_t *) ptr, size);
                context->face_chunk_idx++;
                int iters = (128*128 + DATA_CHUNK_SIZE-1) / DATA_CHUNK_SIZE;
                if(context->face_chunk_idx >= iters)
                {
                    context->face_chunk_idx = 0;
                    PRINTF("Face photo transfer finished\n");
                }
                PRINTF("Face photo sent (%d bytes)\n", size);
            }
            else
            {
                PRINTF("ERROR: Empty respond sent\n");
                pi_nina_b112_send_data_blocking(context->ble,&empty_response, 1);
            }
            break;
        case BLE_GET_DESCRIPTOR:
            PRINTF("BLE_GET_DESCRIPTOR request got\n");
            if(context->read_mode && (context->queue_head != context->queue_tail)) // we are reading and have something in queue
            {
                pi_nina_b112_send_data_blocking(context->ble,(uint8_t *) context->l2_strangers[context->queue_head].descriptor, 512*sizeof(short));
                PRINTF("Face descriptor sent for %s\n", context->l2_strangers[context->queue_head].name);
            }
            else
            {
                PRINTF("ERROR: Empty respond sent\n");
                pi_nina_b112_send_data_blocking(context->ble, &empty_response, 1);
            }
            break;
        case BLE_REMOVE:
            PRINTF("BLE_REMOVE request got\n");
            if(context->read_mode && (context->queue_head != context->queue_tail))
            {
                context->read_mode = 0;
                context->queue_head++;
                PRINTF("Queue head: %d, Queue tail: %d\n", context->queue_head, context->queue_tail);
                pi_nina_b112_send_data_blocking(context->ble, &ack, 1);
                PRINTF("BLE_ACK responded\n");
            }
            else
            {
                pi_nina_b112_send_data_blocking(context->ble, &empty_response, 1);
                PRINTF("ERROR: Empty respond sent\n");
            }
            break;

        case BLE_WRITE:
            PRINTF("BLE_WRITE request got\n");
            context->write_mode = 1;
            pi_nina_b112_send_data_blocking(context->ble, &ack, 1);
            PRINTF("BLE_ACK responded\n");
            break;
        case BLE_SET_NAME:
            pi_nina_b112_get_data_blocking(context->ble, (uint8_t *) context->current_name, 16);
            context->current_name[15] = '\0';
            PRINTF("Name %s got\n", context->current_name);

            pi_nina_b112_send_data_blocking(context->ble, &ack, 1);
            PRINTF("BLE_ACK responded\n");
            if (context->write_mode && (context->queue_head != context->queue_tail))
            {
                memcpy(context->l2_strangers[context->queue_head].name, context->current_name, 16);
            }
            break;
        case BLE_SET_DESCRIPTOR:
        {
            // In GAP side, you don't need to devide it into package size,
            // you can program the udma for 1K, than the uDMA will wait for each package
            pi_nina_b112_get_data_blocking(context->ble, (uint8_t *) context->current_descriptor, 512*sizeof(short));
            PRINTF("BLE_SET_DESCRIPTOR request got\n");
            PRINTF("Got face descriptor\n");

            // Add to Known People DB here
            add_to_db(context->current_descriptor, context->current_name);

            pi_nina_b112_send_data_blocking(context->ble, &ack, 1);
            PRINTF("BLE_ACK responded\n");
        } break;

        case BLE_EXIT:
            PRINTF("BLE_EXIT request got\n");
            pi_nina_b112_send_data_blocking(context->ble, &ack, 1);
            PRINTF("Closing BLE connection\n");
            draw_text(context->display, "Client disconnected", LCD_TXT_POS_X, LCD_TXT_POS_Y, 2);

            ble_exit = 1;
            break;
        default:
            PRINTF("Error: invalid request %d\n", action);
    }
}

static void timeout_handler(void *params)
{
    BleContext *context = (BleContext *)params;

    PRINTF("BLE timeout\n");
    draw_text(context->display, "BLE connection lost", LCD_TXT_POS_X, LCD_TXT_POS_Y, 2);

    ble_exit = 1;
}

void admin_body(struct pi_device *display, struct pi_device* gpio_port, uint8_t button_pin)
{
    PRINTF("Starting Admin (BLE) body\n");

    uint8_t rx_buffer[PI_AT_RESP_ARRAY_LENGTH];

    BleContext context;

    context.read_mode = 0;
    context.write_mode = 0;
    context.face_chunk_idx = 0;

    context.display = display;

    clear_stripe(display, LCD_OFF_Y, LCD_HEIGHT); // clear whole screen except the logo
    draw_text(display, "Loading Photos", LCD_TXT_POS_X, LCD_TXT_POS_Y, 2);

    context.queue_head = 0;
    context.queue_tail = getStrangersCount(); // to allocate memory in future
    PRINTF("Found %d strangers in queue\n", context.queue_tail);
    context.current_descriptor = memory_pool;
    context.current_name = memory_pool + 512;
    context.l2_strangers = (Stranger*) (memory_pool + 512 + 16/sizeof(short));
    Stranger* current_stranger = context.l2_strangers;
    char* previews = (char*) &context.l2_strangers[context.queue_tail+1]; // right after the last structure

    PRINTF("Getting the first stranger from queue\n");

    for(int i = 0; i < context.queue_tail; i++)
    {
        context.l2_strangers[i].preview = previews + i*128*128;
        getStranger(i, &context.l2_strangers[i]);
    }

    PRINTF("Switching to UART mode\n");
    draw_text(display, "Enabling BLE", LCD_TXT_POS_X, LCD_TXT_POS_Y, 2);

    pi_pad_set_function(CONFIG_HYPERBUS_DATA6_PAD, CONFIG_UART_RX_PAD_FUNC);

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

    pi_nina_ble_t ble;
    if (pi_nina_b112_open(&ble))
    {
        PRINTF("Failed to open NINA BLE\n");
        return;
    }

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

    rt_time_wait_us(1*1000*1000);
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
    pi_nina_b112_AT_send(&ble, "+UBTLN=" BLE_NAME);
    PRINTF("Set UBTLN\n");
    pi_nina_b112_AT_query(&ble, "+UMRS?", (char *) rx_buffer);
    PRINTF("BLE configuration : %s\n", rx_buffer);
    pi_nina_b112_AT_query(&ble, "+UBTLN?", (char *) rx_buffer);
    PRINTF("BLE name : %s\n", rx_buffer);

    PRINTF("AT Config Done\n");

    draw_text(display, "Waiting for client", LCD_TXT_POS_X, LCD_TXT_POS_Y, 2);

    // Wait for a connection event: +UUBTACLC:<peer handle,0,<remote BT address> or +UUDPC.
    while (1)
    {
		pi_nina_b112_wait_for_event(&ble, rx_buffer);
		PRINTF("Received Event: %s\n", rx_buffer);

		if ((strncmp(rx_buffer, "+UUBTACLC", 9) == 0) ||
		    (strncmp(rx_buffer, "+UUDPC", 6) == 0))
			break;
    }

    // Enter Data Mode
    pi_nina_b112_AT_send(&ble, "O");
    PRINTF("Data Mode Entered!\n");

    draw_text(display, "Client connected", LCD_TXT_POS_X, LCD_TXT_POS_Y, 2);

    // 50 ms delay is required after entering data mode
    #ifdef __FREERTOS__
    vTaskDelay( 50 / portTICK_PERIOD_MS );
    #else
    rt_time_wait_us(50 * 1000);
    #endif

    context.ble = &ble;
    ble_exit = 0;

    struct pi_task ble_command_task;
    if (rt_timer_create(&ble_timer, RT_TIMER_ONE_SHOT, rt_event_get(NULL, timeout_handler, &context)))
    {
        PRINTF("Failed to create timer\n");
    }

    pi_gpio_pin_notif_clear(gpio_port, button_pin);

    while(!ble_exit)
    {
        if(pi_gpio_pin_notif_get(gpio_port, button_pin) != 0)
        {
            PRINTF("Button pressed. Exiting\n");
            pi_gpio_pin_notif_clear(gpio_port, button_pin);
            ble_exit = 1;
        }
        else
        {
            rt_timer_start(&ble_timer, BLE_TIMEOUT);
            pi_nina_b112_get_data(&ble, &action, 1, pi_task_callback(&ble_command_task, ble_protocol_handler, &context));
            rt_event_yield(NULL);
        }
    }

    rt_timer_stop(&ble_timer);
    rt_timer_destroy(&ble_timer);

    // Exit BLE data mode
    rt_time_wait_us(1000 * 1000);
    pi_nina_b112_exit_data_mode(&ble);
    rt_time_wait_us(1000 * 1000);

    PRINTF("Disabling BLE\n");
    pi_nina_b112_close(&ble);

    rt_gpio_set_pin_value(0, GPIOA21_NINA17, 1);
    rt_gpio_set_pin_value(0, GPIOA2_NINA_RST, 0);

    PRINTF("Switching back to HYPERRAM mode\n");
    pi_pad_set_function(CONFIG_HYPERBUS_DATA6_PAD, CONFIG_HYPERRAM_DATA6_PAD_FUNC);

    PRINTF("Dropping strangers info fro L3\n");
    dropStrangers();

    clear_stripe(display, LCD_TXT_POS_Y, LCD_TXT_HEIGHT(2));
    PRINTF("Exiting admin (BLE) mode\n");
}

uint32_t preview_hyper;

int initHandler(struct pi_device * gpio_port)
{
    PRINTF("Setting button handler..\n");
    if(pi_gpio_pin_configure(gpio_port, BUTTON_PIN_ID, PI_GPIO_INPUT))
    {
        PRINTF("Error: cannot configure pin\n");
        return 0;
    }
    pi_gpio_pin_notif_configure(gpio_port, BUTTON_PIN_ID, PI_GPIO_NOTIF_FALL);

    pi_ram_alloc(&HyperRam, &preview_hyper, 128*128);

    PRINTF("Setting button handler..done\n");

    return !0;
}

int prepareStranger(void* preview)
{
    int iterations = 128*128 / 1024;
    for(int i = 0; i < iterations; i++)
    {
        pi_ram_write(&HyperRam, preview_hyper+i*1024, ((char*)preview) + i*1024, 1024);
    }

    return 0;
}

int handleStranger(short* descriptor)
{
    int status = addStrangerL3(preview_hyper, descriptor);
    if (status == 0)
    {
        pi_ram_alloc(&HyperRam, &preview_hyper, 128*128);
    }
    return status;
}

#undef MIN
