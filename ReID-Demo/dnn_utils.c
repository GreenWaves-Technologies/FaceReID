#if defined(__FREERTOS__)
#include "pmsis.h"
#include "pmsis_l2_malloc.h"
#include "drivers/hyperbus.h"
#include "hyperbus_cl_internal.h"
#endif

#ifdef __EMUL__
# include "extra_emul_stubs.h"
#endif

#include "pmsis.h"
#include "param_layer_struct.h"
#include "dnn_utils.h"

short memory_pool[MEMORY_POOL_SIZE];
struct pi_device HyperRam;


int loadLayerFromFsToL2(struct pi_device *fs, const char* file_name, void* buffer, int size)
{
    PRINTF("Loading layer \"%s\" from FS to L2\n", file_name);
    pi_fs_file_t * file = pi_fs_open(fs, file_name, 0);
    if (file == NULL)
    {
        PRINTF("file open failed\n");
        return 0;
    }

    if((int)file->size > size)
    {
        PRINTF("Provided buffer size %d is smaller than file size %d\n", size, file->size);
        return -1;
    }

    pi_task_t task;
    int size_read = pi_fs_read_async(file, buffer, file->size, pi_task_block(&task));
    pi_task_wait_on(&task);
    PRINTF("Read %d bytes from %s\n", size_read, file_name);

    pi_fs_close(file);

    return size_read;
}

void* loadLayerFromFsToL3(struct pi_device *fs, const char* file_name, struct pi_device *hyper, int* layer_size)
{
    signed char* buff = (signed char*)memory_pool;
    PRINTF("Loading layer \"%s\" from FS to L3\n", file_name);

    pi_fs_file_t * file = pi_fs_open(fs, file_name, 0);
    if (file == NULL)
    {
        PRINTF("file open failed\n");
        return NULL;
    }
    uint32_t hyper_buff;
    pi_ram_alloc(hyper, &hyper_buff, file->size);
    if(hyper_buff == NULL)
    {
        PRINTF("HyperRAM allocation failed\n");
        return NULL;
    }

    unsigned int size_total = 0;
    unsigned int size = 0;
    pi_task_t task;
    do
    {
        //PRINTF("Readning data to local bufer\n");
        size = pi_fs_read_async(file, buff, IO_BUFF_SIZE, pi_task_block(&task));
        pi_task_wait_on(&task);
        //PRINTF("Read %d bytes from %s\n", size, file_name);
        size = ((size + 3) & ~3);
        if(size)
        {
            //PRINTF("Writing data to L3\n");
            pi_ram_write(hyper, (uint32_t)(hyper_buff+size_total), buff, size);
            // PRINTF("Writing data to L3 done\n");
        }
        size_total += size;
    } while(size_total < file->size);


    pi_fs_close(file);

    *layer_size = size_total;

    return hyper_buff;
}

void loadLayerFromL3ToL2(struct pi_device *hyper, void* hyper_buff, void* base_addr, int layer_size)
{
    pi_cl_ram_req_t req;
    //PRINTF("hyper_buff address: %p\n", hyper_buff);
    //PRINTF("base_addr: %p, size %d\n", base_addr, layer_size);
    pi_cl_ram_read(hyper, (uint32_t)hyper_buff, base_addr, layer_size, &req);
    //PRINTF("after pi_cl_hyper_read\n");
    pi_cl_ram_read_wait(&req);
    //PRINTF("after pi_cl_hyper_read_wait\n");
}

int get_activations_size(int idx)
{
    int out_width = convLayers[idx].win;
    int out_height = convLayers[idx].hin;

    if(!convLayers[idx].conv_padding)
    {
        out_width = out_width - convLayers[idx].kernel_width + 1;
        out_height = out_height - convLayers[idx].kernel_height + 1;
    }

    out_width = out_width / convLayers[idx].conv_stride;
    out_height = out_height / convLayers[idx].conv_stride;

    // see output size formulae at https://pytorch.org/docs/0.4.0/nn.html#torch.nn.MaxPool2d
    // dilation = 1, padding = 0
    if(convLayers[idx].max_pool)
    {
        out_width = (1.f*(out_width-(convLayers[idx].pool_size-1) - 1)) / convLayers[idx].pool_stride + 1;
        out_height = (1.f*(out_height-(convLayers[idx].pool_size-1) - 1)) / convLayers[idx].pool_stride + 1;
    }

    int activation_size = convLayers[idx].nb_of * out_height * out_width;

//     PRINTF("Output size for layer %d: %dx%d\n", idx, out_width, out_height);
//     PRINTF("activation_size %d: %d\n", idx, activation_size);

    return activation_size;
}

unsigned int l2_distance(short* v1, short* v2)
{
    unsigned int sum = 0;

    for (int i = 0; i < FACE_DESCRIPTOR_SIZE; i++)
    {
        int delta = v1[i]-v2[i];
        sum += delta*delta;
    }

    return sum;
}
