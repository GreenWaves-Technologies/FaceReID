#ifndef CCN_PULP
#include <stdio.h>
#include <stdint.h>
#endif

#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "pmsis.h"
#include "bsp/fs.h"
#include "bsp/flash/hyperflash.h"


#include "param_layer_struct.h"

#if defined(__FREERTOS__)
# include "pmsis_l2_malloc.h"
# include "pmsis_driver_core_api.h"
# include "pmsis_task.h"
# include "pmsis_os.h"
# include "drivers/hyperbus.h"
# include "hyperbus_cl_internal.h"
# include "pmsis_tiling.h"
#else
# include "Gap.h"
#endif

#define IMAGE_WIDTH 128
#define IMAGE_HEIGHT 128

#include "setup.h"

#include "ImgIO.h"
#include "network_process_manual.h"
#include "dnn_utils.h"

short* infer_result;
short * l2_x;

#ifdef PGM_INPUT
L2_MEM unsigned char tmp_buffer[IMAGE_WIDTH*IMAGE_HEIGHT];
#endif

int activation_size = 0;

static void cluster_main()
{
    infer_result = network_process(&activation_size);
}

void body(void* parameters)
{
    (void) parameters;
    struct pi_device cluster_dev;
    struct pi_cluster_conf cluster_conf;
    struct pi_cluster_task cluster_task;
    cluster_task.stack_size = CLUSTER_STACK_SIZE;
    struct pi_hyper_conf hyper_conf;
    int File = 0;

    PRINTF("main call\n");

    pi_hyperram_conf_init(&hyper_conf);
    pi_open_from_conf(&HyperRam, &hyper_conf);

    if (pi_ram_open(&HyperRam))
    {
        PRINTF("Error: cannot open Hyperram!\n");
        pmsis_exit(-2);
    }

    PRINTF("HyperRAM config done\n");

    PRINTF("HyperRAM config done\n");

    // The hyper chip need to wait a bit.
    // TODO: find out need to wait how many times.
    pi_time_wait_us(1*1000*1000);

    PRINTF("Configuring Hyperflash and FS..\n");
    struct pi_device fs;
    struct pi_device flash;
    struct pi_fs_conf conf;
    struct pi_hyperflash_conf flash_conf;
    pi_fs_conf_init(&conf);

    pi_hyperflash_conf_init(&flash_conf);
    pi_open_from_conf(&flash, &flash_conf);

    if (pi_flash_open(&flash))
    {
        PRINTF("Error: Flash open failed\n");
        pmsis_exit(-3);
    }
    conf.flash = &flash;

    pi_open_from_conf(&fs, &conf);

    int error = pi_fs_mount(&fs);
    if (error)
    {
        PRINTF("Error: FS mount failed with error %d\n", error);
        pmsis_exit(-3);
    }

    PRINTF("FS mounted\n");

    PRINTF("Loading layers to HyperRAM\n");
    network_load(&fs);

    PRINTF("Unmount FS as it's not needed any more\n");
    pi_fs_unmount(&fs);
    PRINTF("FS unmounted\n");

#ifdef PGM_INPUT
    char *inputBlob = "../../../input.pgm";
#else
    char *inputBlob = "../../../input.bin";
#endif
    char *outputBlob = "../../../output.bin";

#if !defined(__FREERTOS__)
    rt_event_sched_t sched;
    rt_event_sched_init(&sched);
    if (rt_event_alloc(&sched, 4)) pmsis_exit(-4);
#endif

    PRINTF("Init cluster...\n");
    pi_cluster_conf_init(&cluster_conf);
    cluster_conf.id = 0;
    cluster_conf.device_type = 0;
    pi_open_from_conf(&cluster_dev, &cluster_conf);
    PRINTF("before pi_cluster_open\n");
    pi_cluster_open(&cluster_dev);
    PRINTF("Init cluster...done\n");

#if !defined(__FREERTOS__)
    //Setting FC to 250MHz
    rt_freq_set(RT_FREQ_DOMAIN_FC, 250000000);

    //Setting Cluster to 150MHz
    // NOTE: Current Gap8 generation does not have clock divider for hyperbus
    // and using FC clocks over 150Mhz is dengerous
    rt_freq_set(RT_FREQ_DOMAIN_CL, 175000000);
#endif

    l2_x = network_init();
    PRINTF("Network init done\n");

    PRINTF("Reading input from host...\n");
    rt_bridge_connect(1, NULL);
    PRINTF("rt_bridge_connect done\n");

#ifdef PGM_INPUT
    int input_size = IMAGE_WIDTH*IMAGE_HEIGHT;
    unsigned int Wi = IMAGE_WIDTH;
    unsigned int Hi = IMAGE_HEIGHT;
    PRINTF("Reading PGM\n");
    char* tmp_buffer2 = ReadImageFromFile(inputBlob, &Wi, &Hi, tmp_buffer, input_size);
    if(tmp_buffer != tmp_buffer2)
    {
        PRINTF("Failed to read PGM image %dx%d\n", Wi, Hi);
        pmsis_exit(-5);
    }

    for(int i = 0; i < input_size; i++)
    {
        l2_x[i] = tmp_buffer[i];
    }

    PRINTF("Writing input.bin\n");
    File = rt_bridge_open("../../../input.bin", O_RDWR | O_CREAT, S_IRWXU, NULL);
    if (File == 0)
    {
        PRINTF("Failed to open file, %s\n", inputBlob);
        pmsis_exit(-6);
    }

    rt_bridge_write(File, l2_x, input_size*sizeof(short), NULL);
    rt_bridge_close(File, NULL);
#else
    File = rt_bridge_open(inputBlob, 0, 0, NULL);
    if (File == 0)
    {
        PRINTF("Failed to open file, %s\n", inputBlob);
        pmsis_exit(-7);
    }
    int input_size = IMAGE_WIDTH*IMAGE_HEIGHT*sizeof(short);
    int read = rt_bridge_read(File, l2_x, input_size, NULL);
    if(read != input_size)
    {
        PRINTF("Failed to read file %s\n", inputBlob);
        PRINTF("Expected input size %d, but read %d\n", input_size, read);
        pmsis_exit(-8);
    }
    rt_bridge_close(File, NULL);
#endif

    PRINTF("DNN inference\n");
#ifdef PERF_COUNT
    unsigned int tm = rt_time_get_us();
#endif
    pi_cluster_send_task_to_cl(&cluster_dev, pi_cluster_task(&cluster_task, (void (*)(void *))cluster_main, NULL));

#ifdef PERF_COUNT
    tm = rt_time_get_us() - tm;
    PRINTF("DNN inference finished in %d microseconds\n", tm);
#endif
    PRINTF("Activations size, shorts: %d\n", activation_size);

    pi_cluster_close(&cluster_dev);

    File = rt_bridge_open(outputBlob, O_RDWR | O_CREAT, S_IRWXU, NULL);
    if (File == 0)
    {
        PRINTF("Failed to open file, %s\n", inputBlob);
        pmsis_exit(-9);
    }

    rt_bridge_write(File, infer_result, activation_size*sizeof(short), NULL);
    rt_bridge_close(File, NULL);
    rt_bridge_disconnect(NULL);

    network_free();

    pmsis_exit(0);
}

int main()
{
    PRINTF("Start First-n-Layers Test\n");
    pmsis_kickoff(body);
    return 0;
}
