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

#if !defined(__FREERTOS__)
#include "Gap.h"
#endif

#define CASCADE_STAGES_L1 5
#define CASCADE_TOTAL_STAGES 25

#define ENABLE_LAYER_1
#define ENABLE_LAYER_2
#define ENABLE_LAYER_3

#define DETECT_STRIDE 1

#define NON_MAX_THRES 250

#define WOUT_INIT 64
#define HOUT_INIT 48


typedef struct single_cascade
{
    //unsigned short num_stages;
    unsigned short stage_size;

    unsigned short rectangles_size;
    short* thresholds;
    short* alpha1;
    short* alpha2;
    unsigned short*  rect_num;
    signed char*  weights;
    char*  rectangles;
} single_cascade_t;

typedef struct cascade
{
    int stages_num;              //number of cascades
    signed short *thresholds;    //cascades thresholds
    single_cascade_t ** stages ;  //pointer to single cascade stages
    single_cascade_t* buffers_l1[2];
} cascade_t;

typedef struct cascade_answers
{
    int x;
    int y;
    int w;
    int h;
    int score;
    char layer_idx;
} cascade_reponse_t;

typedef struct ArgCluster
{
    unsigned char* ImageIn;
    unsigned char* OutCamera;
    unsigned int Win;
    unsigned int Hin;
    unsigned char* ImageOut;
    unsigned int Wout;
    unsigned int Hout;
    unsigned int* ImageIntegral;
    unsigned int* SquaredImageIntegral;
    unsigned char * ImageRender;
    cascade_reponse_t* reponses;
    unsigned char num_reponse;
    int* output_map;
    cascade_t* model;
    unsigned int cycles;
#ifdef PERF_COUNT
    rt_perf_t *perf;
#endif
} ArgCluster_T;

cascade_t *getFaceCascade();
int biggest_cascade_stage(cascade_t *cascade);
void cascade_detect(ArgCluster_T *ArgC);

void detection_cluster_init(ArgCluster_T *ArgC);
void detection_cluster_main(ArgCluster_T *ArgC);

#endif
