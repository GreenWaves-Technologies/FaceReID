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

#include "pmsis.h"
#include "Gap.h"

#include "facedet_pipeline.h"
#include "FaceDetKernels.h"
#include "setup.h"

#include "ImageDraw.h"

static inline unsigned int __attribute__((always_inline)) ChunkSize(unsigned int X)
{
    unsigned int NCore;
    unsigned int Log2Core;
    unsigned int Chunk;

    NCore = gap_ncore();
    Log2Core = gap_fl1(NCore);
    Chunk = (X>>Log2Core) + ((X&(NCore-1))!=0);
    return Chunk;
}

void detection_cluster_init(ArgCluster_T *ArgC)
{
    // PRINTF ("Cluster Init start\n");
    FaceDet_L1_Memory = pi_l1_malloc(ArgC->cl, _FaceDet_L1_Memory_SIZE);
    if (FaceDet_L1_Memory == NULL)
    {
        PRINTF("Failed to allocate %d bytes for L1_memory\n", _FaceDet_L1_Memory_SIZE);
        return;
    }

    //Get Cascade Model
    ArgC->model = getFaceCascade(ArgC->cl);
}

static void prepare_to_render(ArgCluster_T *ArgC)
{
    unsigned int width = ArgC->Win;
    unsigned int height = ArgC->Hin;

    unsigned int CoreId = pi_core_id();
    unsigned int ChunkCell = ChunkSize(height);
    unsigned int First = CoreId*ChunkCell, Last = Min(height, First+ChunkCell);

    for(unsigned int i = First/2; i < Last; i++)
    {
        for(unsigned int j = 0; j < width/2; j++)
        {
            ArgC->ImageRender[((i/2)*(width/2))+j] = ArgC->ImageIn[((i)*width)+2*j];
        }
    }
}

static void draw_response(unsigned char *ImageIn, int Win, int Hin, const cascade_response_t *response)
{
    if (response->score > 0)
    {
        DrawRectangle(ImageIn, Win, Hin, response->x, response->y, response->w, response->h, 0);
        DrawRectangle(ImageIn, Win, Hin, response->x-1, response->y-1, response->w+2, response->h+2, 255);
        DrawRectangle(ImageIn, Win, Hin, response->x-2, response->y-2, response->w+4, response->h+4, 255);
        DrawRectangle(ImageIn, Win, Hin, response->x-3, response->y-3, response->w+6, response->h+6, 0);

        PRINTF("Found face at (%d,%d) with size (%d,%d) at scale %d\n", response->x, response->y, response->w, response->h, response->layer_idx);
    }
}

void detection_cluster_main(ArgCluster_T *ArgC)
{
    #ifdef PERF_COUNT
    gap_cl_starttimer();
    gap_cl_resethwtimer();
    unsigned int Ta = gap_cl_readhwtimer();
    #endif

    cascade_detect(ArgC);

    #ifdef PERF_COUNT
    ArgC->cycles = gap_cl_readhwtimer() - Ta;
    #endif

    draw_response(ArgC->ImageIn, ArgC->Win, ArgC->Hin, ArgC->response);

    //Converting image to RGB 565 for LCD screen and binning image to half the size
    pi_cl_team_fork(gap_ncore(), (void *)prepare_to_render, ArgC);
}

static int check_intersection(const cascade_response_t *a, const cascade_response_t *b)
{
    if ((a->x + a->w - 1 <= b->x) || (b->x + b->w - 1 <= a->x) ||
        (a->y + a->h - 1 <= b->y) || (b->y + b->h - 1 <= a->y))
    {
        return 0;
    }

    return 1;
}

int is_detection_stable(const cascade_response_t *history, int history_size)
{
    for (int i = 0; i < history_size - 1; i++)
    {
        if (check_intersection(&history[i], &history[i+1]) == 0)
        {
            return 0;
        }
    }

    return 1;
}
