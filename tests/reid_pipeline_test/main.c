#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <limits.h>

# include "pmsis.h"

#if defined(__FREERTOS__)
# include "pmsis_l2_malloc.h"
# include "pmsis_driver_core_api.h"
# include "pmsis_task.h"
# include "pmsis_os.h"
# include "drivers/hyperbus.h"
# include "hyperbus_cl_internal.h"
# include "pmsis_tiling.h"
#endif

#include "bsp/bsp.h"
#include "bsp/fs.h"
#include "bsp/flash/hyperflash.h"

#include "bsp/gapoc_a.h"

#include "ImgIO.h"

#include "cascade.h"
#include "setup.h"

#include "network_process_manual.h"
#include "dnn_utils.h"
#include "face_db.h"

#include "CnnKernels.h"
#include "ExtraKernels.h"
#include "reid_pipeline.h"

char* tmp_frame_buffer = (char*)(memory_pool+MEMORY_POOL_SIZE) - CAMERA_WIDTH*CAMERA_HEIGHT;
// Largest possible face after Cascade
char* tmp_face_buffer = (char*)(memory_pool+MEMORY_POOL_SIZE) - CAMERA_WIDTH*CAMERA_HEIGHT - 194*194;
char* tmp_img_face_buffer = (char*)(memory_pool+MEMORY_POOL_SIZE) - CAMERA_WIDTH*CAMERA_HEIGHT - 194*194-128*128;

#if defined(_FOR_GAPOC_)
char *inputBlob = "../../../input_320x240.pgm";
L2_MEM cascade_reponse_t test_response =
{
    .x = 96,
    .y = 56,
    .w = 128,
    .h = 128,
    .score = 1,
    .layer_idx = 0,
};
#else
char *inputBlob = "../../../input_324x244.pgm";
L2_MEM cascade_reponse_t test_response =
{
    .x = 98,
    .y = 58,
    .w = 128,
    .h = 128,
    .score = 1,
    .layer_idx = 0,
};
#endif

char *outputImage = "../../../output.pgm";
char *outputBlob = "../../../output.bin";

// L2_MEM cascade_reponse_t test_response =
// {
//     .x = 113,
//     .y = 97,
//     .w = 121,
//     .h = 121,
//     .score = 1,
//     .layer_idx = 0,
// };

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

void body(void * parameters)
{
    (void) parameters;
    struct pi_device cluster_dev;
    struct pi_cluster_conf cluster_conf;
    struct pi_cluster_task cluster_task;
    struct pi_hyper_conf hyper_conf;
    cluster_task.stack_size = CLUSTER_STACK_SIZE;

    PRINTF("Start ReID Pipeline test\n");

    pi_hyperram_conf_init(&hyper_conf);
    pi_open_from_conf(&HyperRam, &hyper_conf);

    if (pi_ram_open(&HyperRam))
    {
        PRINTF("Error: cannot open Hyperram!\n");
        pmsis_exit(-2);
    }

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

    if (int error = pi_fs_mount(&fs))
    {
        PRINTF("Error: FS mount failed with error %d\n", error);
        pmsis_exit(-3);
    }

    PRINTF("FS mounted\n");

    PRINTF("Loading layers to HyperRAM\n");
    network_load(&fs);

    PRINTF("Loading static ReID database\n");
    load_static_db(&fs);

    PRINTF("Unmount FS as it's not needed any more\n");
    pi_fs_unmount(&fs);
    PRINTF("Unmount FS done\n");

    PRINTF("Reading image from host...\n");
    rt_bridge_connect(1, NULL);
    PRINTF("rt_bridge_connect done\n");

    int input_size = CAMERA_WIDTH*CAMERA_HEIGHT;
    unsigned int Wi = CAMERA_WIDTH;
    unsigned int Hi = CAMERA_HEIGHT;

    PRINTF("Before ReadImageFromFile\n");
    int read = ReadImageFromFile(inputBlob, &Wi, &Hi, tmp_frame_buffer, input_size);
    PRINTF("After ReadImageFromFile with status: %d\n", read);
    if(!read)
    {
        PRINTF("Failed\n");
        pmsis_exit(-4);
    }
    PRINTF("Host file read\n");

    PRINTF("Init cluster...\n");
    pi_cluster_conf_init(&cluster_conf);
    cluster_conf.id = 0;
    cluster_conf.device_type = 0;
    pi_open_from_conf(&cluster_dev, &cluster_conf);
    PRINTF("before pi_cluster_open\n");
    pi_cluster_open(&cluster_dev);
    PRINTF("Init cluster...done\n");

    ArgClusterDnn_T ClusterDnnCall;
    ClusterDnnCall.roi         = &test_response;
    ClusterDnnCall.frame       = tmp_frame_buffer;
    ClusterDnnCall.face        = tmp_face_buffer;
    ClusterDnnCall.scaled_face = network_init();
    if(!ClusterDnnCall.scaled_face)
    {
        PRINTF("Failed to initialize ReID network!\n");
        pmsis_exit(-6);
    }

    ExtaKernels_L1_Memory = L1_Memory;

#ifdef PERF_COUNT
    unsigned int tm = rt_time_get_us();
#endif
    PRINTF("Before pi_cluster_send_task_to_cl 1\n");
    pi_cluster_send_task_to_cl(&cluster_dev, pi_cluster_task(&cluster_task, (void (*)(void *))reid_prepare_cluster, &ClusterDnnCall));
    PRINTF("After pi_cluster_send_task_to_cl 1\n");

    my_copy(ClusterDnnCall.scaled_face, tmp_img_face_buffer, 128, 128);

    WriteImageToFile(outputImage, 128, 128, tmp_img_face_buffer);

    PRINTF("Before pi_cluster_send_task_to_cl 2\n");
    pi_cluster_send_task_to_cl(&cluster_dev, pi_cluster_task(&cluster_task, (void (*)(void *))reid_inference_cluster, &ClusterDnnCall));
    PRINTF("After pi_cluster_send_task_to_cl 2\n");

    int File = rt_bridge_open(outputBlob, O_RDWR | O_CREAT, S_IRWXU, NULL);
    if (File == 0)
    {
        PRINTF("Failed to open file, %s\n", inputBlob);
        pmsis_exit(-7);
    }

    rt_bridge_write(File, ClusterDnnCall.output, ClusterDnnCall.activation_size*sizeof(short), NULL);
    rt_bridge_close(File, NULL);
    rt_bridge_disconnect(NULL);

    char* person_name;
    int id_conf = identify_by_db(ClusterDnnCall.output, &person_name);
    PRINTF("Hi, %s! Conf: %d\n", person_name, id_conf);

#ifdef PERF_COUNT
    tm = rt_time_get_us() - tm;
    PRINTF("Cycle time %d microseconds\n", tm);
#endif

    // Close the cluster
    pi_cluster_close(&cluster_dev);

    pmsis_exit(0);
}

int main()
{
    PRINTF("Start full ReID pipeline Test\n");
    pmsis_kickoff(body);
    return 0;
}
