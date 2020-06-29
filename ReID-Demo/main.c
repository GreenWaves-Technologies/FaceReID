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

#include <stdio.h>
#include <limits.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "pmsis.h"
#include "bsp/flash/hyperflash.h"

#if defined(__FREERTOS__)
# include "pmsis_driver_core_api.h"
# include "pmsis_task.h"
# include "pmsis_os.h"
# include "drivers/hyperbus.h"
# include "hyperbus_cl_internal.h"
# include "pmsis_tiling.h"
#else
# include "Gap.h"
#endif

#include "bsp/fs.h"
#include "bsp/fs/hostfs.h"
#include "bsp/fs/readfs.h"

#include "bsp/bsp.h"
#include "bsp/buffer.h"
#include "bsp/camera/himax.h"
#include "bsp/camera/mt9v034.h"
#include "bsp/ram/hyperram.h"

#include "setup.h"
#include "ImgIO.h"
#include "cascade.h"
#include "display.h"

#include "strangers_db.h"

#if defined(STATIC_FACE_DB)
#  if defined(BLE_NOTIFIER)
#    include "StaticUserManagerBleNotifier.h"
#  else
#    include "StaticUserManager.h"
#  endif
#elif defined(USE_BLE_USER_MANAGEMENT)
#include "BleUserManager.h"
#endif

#include "network_process.h"
#include "dnn_utils.h"
#include "face_db.h"

#include "reid_pipeline.h"
#include "facedet_pipeline.h"

cascade_reponse_t responses[MAX_NUM_OUT_WINS];

static void my_copy(short* in, unsigned char* out, int Wout, int Hout)
{
    for(int i = 0; i < Hout; i++)
    {
        for(int j = 0; j < Hout; j++)
        {
            out[i*Wout + j] = (unsigned char)in[i*Wout + j];
        }
    }
}

#define MT9V034_BLACK_LEVEL_CTRL  0x47
#define MT9V034_BLACK_LEVEL_AUTO  (0 << 0)
#define MT9V034_AEC_AGC_ENABLE    0xaf
#define MT9V034_AEC_ENABLE_A      (1 << 0)
#define MT9V034_AGC_ENABLE_A      (1 << 1)

#if defined(CONFIG_GAPOC_A)
static int open_camera_mt9v034(struct pi_device *device)
{
    uint16_t val;
    struct pi_mt9v034_conf cam_conf;

    pi_mt9v034_conf_init(&cam_conf);

    //cam_conf.column_flip = 1;
    //cam_conf.row_flip    = 0;
    #ifdef QVGA
    cam_conf.format = CAMERA_QVGA;
    #endif
    #ifdef QQVGA
    cam_conf.format = CAMERA_QQVGA;
    #endif

    pi_open_from_conf(device, &cam_conf);
    if (pi_camera_open(device))
        return -1;

    val = MT9V034_BLACK_LEVEL_AUTO;
    pi_camera_reg_set(device, MT9V034_BLACK_LEVEL_CTRL, (uint8_t *) &val);

    val = MT9V034_AEC_ENABLE_A | MT9V034_AGC_ENABLE_A;
    pi_camera_reg_set(device, MT9V034_AEC_AGC_ENABLE, (uint8_t *) &val);

    //MAX and MIN AEC
    //Minimum Coarse Shutter Width
    val = 0x0000; //def 1
    pi_camera_reg_set(device, 0xAC, (uint8_t *) &val);
    //Maximum Coarse Shutter Width
    val = 0x05E0; //def 480 0x01E0
    pi_camera_reg_set(device, 0xAD, (uint8_t *) &val);

    //MAX Analog Gain
    val = 0x40; //def 64 0x40, value 16 to 64
    pi_camera_reg_set(device, 0xAB, (uint8_t *) &val);

    //AGC/AEC Pixel Count
    val =0xABE0; //0-65535, def 44000 0xABE0
    pi_camera_reg_set(device, 0xB0, (uint8_t *) &val);

    //Desired luminance of the image by setting a desired bin
    val = 32; //def 58
    pi_camera_reg_set(device, 0xA5, (uint8_t *) &val);

    return 0;
}
#else
static int open_camera_himax(struct pi_device *device)
{
    struct pi_himax_conf cam_conf;

    pi_himax_conf_init(&cam_conf);

    #ifdef QVGA
    cam_conf.format = CAMERA_QVGA;
    #endif

    pi_open_from_conf(device, &cam_conf);
    if (pi_camera_open(device))
        return -1;

    return 0;
}
#endif

static int open_camera(struct pi_device *device)
{
#if defined(HAVE_CAMERA)
#if defined(CONFIG_GAPOC_A)
    return open_camera_mt9v034(device);
#else
    return open_camera_himax(device);
#endif
    return -1;
#else
    return 0;
#endif
}

#if defined(USE_BLE_USER_MANAGEMENT) || defined(BLE_NOTIFIER)
static int open_gpio(struct pi_device *device)
{
    struct pi_gpio_conf gpio_conf;

    pi_gpio_conf_init(&gpio_conf);
    pi_open_from_conf(device, &gpio_conf);

    if (pi_gpio_open(device))
        return -1;

    pi_gpio_pin_configure(device, GPIOA2_NINA_RST,  PI_GPIO_OUTPUT);
    pi_gpio_pin_configure(device, GPIOA21_NINA17,   PI_GPIO_OUTPUT);

    pi_gpio_pin_write(device, GPIOA2_NINA_RST,  0);
    pi_gpio_pin_write(device, GPIOA21_NINA17,   1);

    return 0;
}
#endif

#define IMAGE_SIZE (CAMERA_WIDTH * CAMERA_HEIGHT)

void body(void* parameters)
{
    (void) parameters;
    static pi_buffer_t RenderBuffer;
    char* person_name;
    struct pi_device cluster_dev;
    struct pi_device gpio_port;
    struct pi_device camera;
    struct pi_device display;
    struct pi_cluster_conf cluster_conf;
    struct pi_cluster_task cluster_task;
    cascade_reponse_t cascade_history[FACE_DETECTOR_STABILIZATION_PERIOD];
    int cascade_history_size = 0;

    unsigned char* ImageRender;
    unsigned char* ImageIn;
    unsigned char* ImageOut;
    unsigned int* ImageIntegral;
    unsigned int* SquaredImageIntegral;
    int* output_map;

    char string_buffer[64];
    ArgCluster_T ClusterDetectionCall;
    ArgClusterDnn_T ClusterDnnCall;

    PRINTF("Start ReID Demo Application\n");

    pi_freq_set(PI_FREQ_DOMAIN_FC, 50000000);

#if defined(USE_BLE_USER_MANAGEMENT) || defined(BLE_NOTIFIER)
    if (open_gpio(&gpio_port))
    {
        PRINTF("Error: cannot open GPIO port\n");
        pmsis_exit(-4);
    }
#endif

#if defined(HAVE_DISPLAY)
    PRINTF("Initializing display\n");
    if (open_display(&display))
    {
        pmsis_exit(-5);
    }
    PRINTF("Initializing display done\n");

    clear_stripe(&display, 0, LCD_HEIGHT);
    setTextColor(&display, LCD_TXT_CLR);
    draw_gwt_logo(&display);
    draw_text(&display, "Loading network", LCD_TXT_POS_X, LCD_TXT_POS_Y, 2);
#endif

    PRINTF("Camera resolution: %dx%d\n", CAMERA_WIDTH, CAMERA_HEIGHT);

    PRINTF("Configuring Hyperram..\n");
    struct pi_hyperram_conf hyper_conf;

    pi_hyperram_conf_init(&hyper_conf);
    pi_open_from_conf(&HyperRam, &hyper_conf);
    if (pi_ram_open(&HyperRam))
    {
        PRINTF("Error: cannot open Hyperram!\n");
        pmsis_exit(-2);
    }

    PRINTF("HyperRAM config done\n");

    PRINTF("Configuring Hyperflash and FS..\n");
    struct pi_device flash;
    struct pi_hyperflash_conf flash_conf;

    pi_hyperflash_conf_init(&flash_conf);
    pi_open_from_conf(&flash, &flash_conf);
    if (pi_flash_open(&flash))
    {
        PRINTF("Error: Flash open failed\n");
        pmsis_exit(-3);
    }

    // The hyper chip needs to wait a bit.
    pi_time_wait_us(100 * 1000);

    struct pi_device fs;
    struct pi_readfs_conf fs_conf;

    pi_readfs_conf_init(&fs_conf);
    fs_conf.fs.flash = &flash;
    pi_open_from_conf(&fs, &fs_conf);

    int error = pi_fs_mount(&fs);
    if (error)
    {
        PRINTF("Error: FS mount failed with error %d\n", error);
        pmsis_exit(-3);
    }

    PRINTF("FS mounted\n");

    PRINTF("Loading layers to HyperRAM\n");
    network_load(&fs);

    int status = 1;
#if defined(STATIC_FACE_DB)
# if defined(BLE_NOTIFIER)
    status = initHandler(&fs, &display);
# else
    status = initHandler(&fs);
# endif
#elif defined(USE_BLE_USER_MANAGEMENT)
    status = initHandler(&gpio_port);
#endif
    if(!status)
    {
        PRINTF("User manager init failed!\n");
        pmsis_exit(-5);
    }

    PRINTF("Unmount FS as it's not needed any more\n");
    pi_fs_unmount(&fs);
    PRINTF("Unmount FS done\n");

#ifdef DUMP_SUCCESSFUL_FRAME
    struct pi_hostfs_conf host_fs_conf;
    pi_hostfs_conf_init(&host_fs_conf);
    struct pi_device host_fs;

    pi_open_from_conf(&host_fs, &host_fs_conf);

    if (pi_fs_mount(&host_fs))
    {
        PRINTF("pi_fs_mount failed\n");
        pmsis_exit(-8);
    }
#endif

    PRINTF("Init cluster...\n");
    pi_cluster_conf_init(&cluster_conf);
    cluster_conf.id = 0;
    cluster_conf.device_type = 0;
    pi_open_from_conf(&cluster_dev, &cluster_conf);
    pi_cluster_open(&cluster_dev);
    PRINTF("Init cluster...done\n");

    //Setting FC to 200MHz
    pi_freq_set(PI_FREQ_DOMAIN_FC, 200000000);

    //Setting Cluster to 150MHz
    // NOTE: Current Gap8 generation does not have clock divider for hyperbus
    // and using FC clocks over 150Mhz is dangerous
    pi_freq_set(PI_FREQ_DOMAIN_CL, 150000000);

    // HACK: Init display for the second time, because
    // SPI API does not handle clocks change correctly for now
#if defined(HAVE_DISPLAY)
    PRINTF("Initializing display for the second time\n");
    if (open_display(&display))
    {
        pmsis_exit(-5);
    }
    clear_stripe(&display, LCD_TXT_POS_Y, LCD_TXT_HEIGHT(2));
    setTextColor(&display, LCD_TXT_CLR);
    PRINTF("Initializing display done\n");
#endif

    // put camera frame to memory pool tail, it does not intersect with the first DNN layer data
    ImageIn = ((unsigned char*)(memory_pool + MEMORY_POOL_SIZE)) - IMAGE_SIZE;
    ImageOut = ImageIn - WOUT_INIT*HOUT_INIT;
    ImageIntegral = ((unsigned int*)ImageOut) - WOUT_INIT*HOUT_INIT;
    SquaredImageIntegral = ImageIntegral - WOUT_INIT*HOUT_INIT;
    output_map = SquaredImageIntegral - (HOUT_INIT-24+1)*(WOUT_INIT-24+1);
    ImageRender = memory_pool;

    pi_buffer_init(&RenderBuffer, PI_BUFFER_TYPE_L2, ImageOut);
    pi_buffer_set_format(&RenderBuffer, CAMERA_WIDTH/2, CAMERA_HEIGHT/2, 1, PI_BUFFER_FORMAT_GRAY);

    PRINTF("Initializing camera\n");
    if (open_camera(&camera))
    {
        PRINTF("Error: Failed to initialize camera\n");
        pmsis_exit(-6);
    }

    ClusterDetectionCall.cl                   = &cluster_dev;
    ClusterDetectionCall.ImageIn              = ImageIn;
    ClusterDetectionCall.Win                  = CAMERA_WIDTH;
    ClusterDetectionCall.Hin                  = CAMERA_HEIGHT;
    ClusterDetectionCall.Wout                 = WOUT_INIT;
    ClusterDetectionCall.Hout                 = HOUT_INIT;
    ClusterDetectionCall.ImageOut             = ImageOut;
    ClusterDetectionCall.ImageIntegral        = ImageIntegral;
    ClusterDetectionCall.SquaredImageIntegral = SquaredImageIntegral;
    ClusterDetectionCall.ImageRender          = ImageRender;
    ClusterDetectionCall.output_map           = output_map;
    ClusterDetectionCall.reponses             = responses;

    //Cluster Init
    pi_cluster_task(&cluster_task, (void (*)(void *))detection_cluster_init, &ClusterDetectionCall);
    cluster_task.slave_stack_size = CL_SLAVE_STACK_SIZE;
    cluster_task.stack_size = CL_STACK_SIZE;
    pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

    PRINTF("Main cycle\n");

    //draw_text(&display, "Ready", LCD_TXT_POS_X, LCD_TXT_POS_Y, 2);

    int saved_index = 0;
    while(1)
    {
#if defined(USE_BLE_USER_MANAGEMENT)
        if(pi_gpio_pin_notif_get(&gpio_port, BUTTON_PIN_ID) != 0)
        {
            pi_gpio_pin_notif_clear(&gpio_port, BUTTON_PIN_ID);
            admin_body(&display, &gpio_port, BUTTON_PIN_ID);
        }
#endif
#ifdef HAVE_CAMERA
        pi_camera_control(&camera, PI_CAMERA_CMD_START, 0);
        pi_camera_capture(&camera, ImageIn, IMAGE_SIZE);
        pi_camera_control(&camera, PI_CAMERA_CMD_STOP, 0);
#endif

#ifdef PERF_COUNT
        unsigned int tm = rt_time_get_us();
#endif
        pi_cluster_task(&cluster_task, (void (*)(void *))detection_cluster_main, &ClusterDetectionCall);
        cluster_task.slave_stack_size = CL_SLAVE_STACK_SIZE;
        cluster_task.stack_size = CL_STACK_SIZE;
        pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

#if defined(HAVE_DISPLAY)
        RenderBuffer.data = ImageRender;
        pi_display_write(&display, &RenderBuffer, LCD_OFF_X, LCD_OFF_Y, CAMERA_WIDTH/2,CAMERA_HEIGHT/2);
#endif

        if(ClusterDetectionCall.num_reponse)
        {
            PRINTF("Faces detected\n");

            int optimal_detection_id = -1;
            int optimal_score = -1;
            for(int i = 0; i < ClusterDetectionCall.num_reponse; i++)
            {
                if(responses[i].score > optimal_score)
                {
                    optimal_detection_id = i;
                    optimal_score = responses[i].score;
                }
            }

            memcpy(&cascade_history[cascade_history_size], &responses[optimal_detection_id], sizeof(cascade_reponse_t));
            cascade_history_size++;

            if(cascade_history_size == FACE_DETECTOR_STABILIZATION_PERIOD)
            {
                if(check_detection_stability(cascade_history, FACE_DETECTOR_STABILIZATION_PERIOD))
                {
                    cascade_history_size = 0;
                    PRINTF("Face detection is stable enough, run ReID\n");

                    // Reset cluster (frees all L1 memory after cascades)
                    pi_cluster_close(&cluster_dev);
                    pi_time_wait_us(10 * 1000);
                    pi_cluster_open(&cluster_dev);

                    ClusterDnnCall.roi         = &responses[optimal_detection_id];
                    ClusterDnnCall.frame       = ImageIn;
                    ClusterDnnCall.face        = ((unsigned char*)output_map) - (194*194); // Largest possible face after Cascade
                    ClusterDnnCall.scaled_face = network_init(&cluster_dev);
                    if(!ClusterDnnCall.scaled_face)
                    {
                        PRINTF("Failed to initialize ReID network!\n");
                        pmsis_exit(-7);
                    }

                    pi_cluster_task(&cluster_task, (void (*)(void *))reid_prepare_cluster, &ClusterDnnCall);
                    cluster_task.slave_stack_size = CL_SLAVE_STACK_SIZE;
                    cluster_task.stack_size = CL_STACK_SIZE;
                    pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

#if defined(DUMP_SUCCESSFUL_FRAME) || defined(USE_BLE_USER_MANAGEMENT)
                    my_copy(ClusterDnnCall.scaled_face, ClusterDnnCall.face, 128, 128);
#endif

#ifdef DUMP_SUCCESSFUL_FRAME
                    draw_text(&display, "Writing photo", LCD_TXT_POS_X, LCD_TXT_POS_Y, 2);

                    sprintf(string_buffer, "../../../dumps/face_%d.pgm", saved_index);
                    WriteImageToFile(string_buffer, 128, 128, ClusterDnnCall.face);

                    draw_text(&display, "Writing descriptor", LCD_TXT_POS_X, LCD_TXT_POS_Y, 2);
#endif

#if defined(DUMP_SUCCESSFUL_FRAME) || defined (USE_BLE_USER_MANAGEMENT)
                    prepareStranger(ClusterDnnCall.face);
#endif

#ifdef PERF_COUNT
                    unsigned int inftm = rt_time_get_us();
#endif
                    pi_cluster_task(&cluster_task, (void (*)(void *))reid_inference_cluster, &ClusterDnnCall);
                    cluster_task.slave_stack_size = CL_SLAVE_STACK_SIZE;
                    cluster_task.stack_size = CL_STACK_SIZE;
                    pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
#ifdef PERF_COUNT
                    inftm = rt_time_get_us() - inftm;
                    PRINTF("DNN inference finished in %d microseconds\n", inftm);
#endif


#ifdef DUMP_SUCCESSFUL_FRAME
                    sprintf(string_buffer, "../../../dumps/face_%d.bin", saved_index);

                    pi_fs_file_t* host_file = pi_fs_open(&host_fs, string_buffer, PI_FS_FLAGS_WRITE);
                    if (!host_file)
                    {
                        PRINTF("Failed to open host file, %s\n", string_buffer);
                        pmsis_exit(-7);
                    }

                    pi_fs_write(host_file, ClusterDnnCall.output, 512*sizeof(short));
                    pi_fs_close(host_file);

                    //sprintf(string_buffer, "frame_%d.pgm", saved_index);
                    //WriteImageToFile(string_buffer, CAMERA_WIDTH, CAMERA_HEIGHT, ImageIn);

                    saved_index++;
#endif

                    int id_l2 = identify_by_db(ClusterDnnCall.output, &person_name);

                    //sprintf(string_buffer, "ReID L2: %d\n", id_l2);
                    sprintf(string_buffer, "ReID NN uW/frame/s: %d\n",(int)(16800.f/(50000000.f/ClusterDnnCall.cycles)));
                    //sprintf(string_buffer, "ReID NN GCycles: %d\n", ClusterDnnCall.cycles/1000000);
                    PRINTF(string_buffer);
                    draw_text(&display, string_buffer, LCD_TXT_POS_X, LCD_TXT_POS_Y, 2);

                    if ((id_l2 >= 0) && (id_l2 < REID_L2_THRESHOLD))
                    {
                        pi_cluster_close(&cluster_dev);
                        sprintf(string_buffer, "Hi, %s!\n", person_name);
                        PRINTF(string_buffer);
                        draw_text(&display, string_buffer, LCD_TXT_POS_X, LCD_TXT_POS_Y - 20, 2);
#if defined(BLE_NOTIFIER)
                        handleUser(person_name);
#endif
                    }
                    else
                    {
                        pi_cluster_close(&cluster_dev);
                        PRINTF("STOP, Stranger!\n");
                        draw_text(&display, "STOP, Stranger!\n", LCD_TXT_POS_X, LCD_TXT_POS_Y - 20, 2);
                        PRINTF("Adding stranger to queue\n");
                        status = handleStranger(ClusterDnnCall.output);
                        switch(status)
                        {
                            case 0:
                                PRINTF("Stranger reported!\n");
                                break;
                            case DB_FULL:
                                PRINTF("No space for Stranger!\n");
                                break;
                            case DUPLICATE_DROPPED:
                                PRINTF("Stranger duplicate, dropped!\n");
                                break;
                            default:
                                PRINTF("Error: code=%d", status);
                                break;
                        }
                    }

                    pi_cluster_close(&cluster_dev);

                    pi_cluster_open(&cluster_dev);

                    pi_cluster_task(&cluster_task, (void (*)(void *))detection_cluster_init, &ClusterDetectionCall);
                    cluster_task.slave_stack_size = CL_SLAVE_STACK_SIZE;
                    cluster_task.stack_size = CL_STACK_SIZE;
                    pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
                }
                else
                {
                    PRINTF("Detection is not stable\n");
                    cascade_history_size--;
                    for(int i = 0; i < FACE_DETECTOR_STABILIZATION_PERIOD-1; i++)
                        memcpy(&cascade_history[i], &cascade_history[i+1], sizeof(cascade_reponse_t));
                }
            }
        }
        else
        {
            cascade_history_size = 0;
        }

#ifdef PERF_COUNT
        tm = rt_time_get_us() - tm;
        PRINTF("Cycle time %d microseconds\n", tm);
#endif
    }

#ifdef HAVE_CAMERA
    pi_camera_close(&camera);
#endif

    pi_cluster_close(&cluster_dev);

#ifdef DUMP_SUCCESSFUL_FRAME
    pi_fs_unmount(&host_fs);
#endif

#if defined(BLE_NOTIFIER)
    closeHandler();
#endif

    pmsis_exit(0);
}

int main()
{
    PRINTF("Start ReID Demo\n");
    pmsis_kickoff(body);
    return 0;
}
