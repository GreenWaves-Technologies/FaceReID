#ifndef __REID_PIPELINE_H__
#define __REID_PIPELINE_H__

#include "cascade.h"

typedef struct ArgClusterDnn
{
    cascade_reponse_t* roi;
    unsigned char* frame;
    unsigned char* face;
    short* scaled_face;
    short* output;
    int activation_size;
    unsigned int cycles;
} ArgClusterDnn_T;

void reid_prepare_cluster(ArgClusterDnn_T* ArgC);
void reid_inference_cluster(ArgClusterDnn_T* ArgC);

#endif
