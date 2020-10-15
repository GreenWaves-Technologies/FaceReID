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

#ifndef CASCADE_H
#define CASCADE_H

#include "setup.h"
#include "Gap.h"

#define CASCADE_STAGES_L1 15
#define CASCADE_TOTAL_STAGES 25

#define ENABLE_LAYER_1
#define ENABLE_LAYER_2
#define ENABLE_LAYER_3

#define DETECT_STRIDE 1

typedef struct
{
    unsigned short stage_size;

    unsigned short rectangles_size;
    short* thresholds;
    short* alpha1;
    short* alpha2;
    unsigned short*  rect_num;
    signed char*  weights;
    char*  rectangles;
} single_cascade_t;

typedef struct
{
    signed short *thresholds;    //cascades thresholds
    single_cascade_t ** stages;  //pointer to single cascade stages
    single_cascade_t* buffers_l1[2];
} cascade_t;

typedef struct
{
    int x;
    int y;
    int w;
    int h;
    int score;
    char layer_idx;
} cascade_response_t;

typedef struct
{
    struct pi_device *cl;
    unsigned char* ImageIn;
    unsigned int Win;
    unsigned int Hin;
    unsigned char* ImageOut;
    unsigned int* ImageIntegral;
    unsigned int* SquaredImageIntegral;
    unsigned char * ImageRender;
    cascade_response_t *response;
    int* output_map;
    cascade_t* model;
    unsigned int cycles;
} ArgCluster_T;

cascade_t *getFaceCascade(struct pi_device *cl);
void cascade_detect(ArgCluster_T *ArgC);

void detection_cluster_init(ArgCluster_T *ArgC);
void detection_cluster_main(ArgCluster_T *ArgC);

#endif
