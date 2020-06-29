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

#if defined(__FREERTOS__)
# include "dma/cl_dma.h"
# include "pmsis_os.h"
# include "pmsis_tiling.h"
#else
# include "Gap.h"
#endif

#include "cascade.h"
#include "setup.h"
#include "face_cascade.h"
#include "FaceDetKernels.h"

#include <stdlib.h>
#include <stdio.h>

static unsigned biggest_cascade_stage(const cascade_t *cascade);

//Permanently Store a cascade stage to L1
single_cascade_t* sync_copy_cascade_stage_to_l1(struct pi_device *cl, single_cascade_t* cascade_l2)
{
    pi_cl_dma_cmd_t DmaR_Evt1;

    single_cascade_t *cascade_l1 = pi_l1_malloc(cl, sizeof(single_cascade_t));
    if (cascade_l1 == NULL) {
        PRINTF("Error: Failed to allocate cascade stage\n");
        return NULL;
    }

    cascade_l1->stage_size = cascade_l2->stage_size;
    cascade_l1->rectangles_size = cascade_l2->rectangles_size;

    cascade_l1->thresholds = pi_l1_malloc(cl, sizeof(short) * cascade_l1->stage_size);
    pi_cl_dma_cmd((uint32_t) cascade_l2->thresholds, (uint32_t) cascade_l1->thresholds, sizeof(short)*cascade_l1->stage_size, PI_CL_DMA_DIR_EXT2LOC, &DmaR_Evt1);
    pi_cl_dma_cmd_wait(&DmaR_Evt1);

    cascade_l1->alpha1     = pi_l1_malloc(cl, sizeof(short) * cascade_l1->stage_size);
    pi_cl_dma_cmd((uint32_t) cascade_l2->alpha1, (uint32_t) cascade_l1->alpha1, sizeof(short)*cascade_l1->stage_size, PI_CL_DMA_DIR_EXT2LOC, &DmaR_Evt1);
    pi_cl_dma_cmd_wait(&DmaR_Evt1);

    cascade_l1->alpha2     = pi_l1_malloc(cl, sizeof(short) * cascade_l1->stage_size);
    pi_cl_dma_cmd((uint32_t) cascade_l2->alpha2, (uint32_t) cascade_l1->alpha2, sizeof(short)*cascade_l1->stage_size, PI_CL_DMA_DIR_EXT2LOC, &DmaR_Evt1);
    pi_cl_dma_cmd_wait(&DmaR_Evt1);

    cascade_l1->rect_num   = pi_l1_malloc(cl, sizeof(unsigned short) * (cascade_l1->stage_size+1));
    pi_cl_dma_cmd((uint32_t) cascade_l2->rect_num, (uint32_t) cascade_l1->rect_num, sizeof(unsigned short)*(cascade_l1->stage_size+1), PI_CL_DMA_DIR_EXT2LOC, &DmaR_Evt1);
    pi_cl_dma_cmd_wait(&DmaR_Evt1);

    cascade_l1->weights    = pi_l1_malloc(cl, sizeof(signed char) * (cascade_l1->rectangles_size/4));
    pi_cl_dma_cmd((uint32_t) cascade_l2->weights, (uint32_t) cascade_l1->weights, sizeof(signed char)*(cascade_l1->rectangles_size/4), PI_CL_DMA_DIR_EXT2LOC, &DmaR_Evt1);
    pi_cl_dma_cmd_wait(&DmaR_Evt1);

    cascade_l1->rectangles = pi_l1_malloc(cl, sizeof(char) * cascade_l1->rectangles_size);
    pi_cl_dma_cmd((uint32_t) cascade_l2->rectangles, (uint32_t) cascade_l1->rectangles, sizeof(char)*cascade_l1->rectangles_size, PI_CL_DMA_DIR_EXT2LOC, &DmaR_Evt1);
    pi_cl_dma_cmd_wait(&DmaR_Evt1);

    if (cascade_l1->thresholds == NULL ||
        cascade_l1->alpha1 == NULL ||
        cascade_l1->alpha2 == NULL ||
        cascade_l1->rect_num == NULL ||
        cascade_l1->weights == NULL ||
        cascade_l1->rectangles == NULL)
    {
        PRINTF("Error: Failed to allocate cascade stage\n");
        return NULL;
    }

    return cascade_l1;
}

cascade_t *getFaceCascade(struct pi_device *cl)
{
    cascade_t *face_cascade = pi_l1_malloc(cl, sizeof(cascade_t));
    if (face_cascade == NULL)
    {
        PRINTF("Error: Failed to allocate cascade");
        return NULL;
    }

    single_cascade_t **model_stages = pi_l1_malloc(cl, sizeof(single_cascade_t*) * CASCADE_TOTAL_STAGES);
    face_cascade->thresholds = pi_l1_malloc(cl, sizeof(signed short) * CASCADE_TOTAL_STAGES);
    if (face_cascade->thresholds == NULL)
    {
        PRINTF("Error: Failed to allocate cascade");
        return NULL;
    }

    for(int a = 0; a < CASCADE_TOTAL_STAGES; a++)
        face_cascade->thresholds[a] = model_thresholds[a];

    switch(CASCADE_TOTAL_STAGES){
        case 25:
                model_stages[24] = &stage_24;
        case 24:
                model_stages[23] = &stage_23;
        case 23:
                model_stages[22] = &stage_22;
        case 22:
                model_stages[21] = &stage_21;
        case 21:
                model_stages[20] = &stage_20;
        case 20:
                model_stages[19] = &stage_19;
        case 19:
                model_stages[18] = &stage_18;
        case 18:
                model_stages[17] = &stage_17;
        case 17:
                model_stages[16] = &stage_16;
        case 16:
                model_stages[15] = &stage_15;
        case 15:
                model_stages[14] = &stage_14;
        case 14:
                model_stages[13] = &stage_13;
        case 13:
                model_stages[12] = &stage_12;
        case 12:
                model_stages[11] = &stage_11;
        case 11:
                model_stages[10] = &stage_10;
        case 10:
                model_stages[9] = &stage_9;
        case 9:
                model_stages[8] = &stage_8;
        case 8:
                model_stages[7] = &stage_7;
        case 7:
                model_stages[6] = &stage_6;
        case 6:
                model_stages[5] = &stage_5;
        case 5:
                model_stages[4] = &stage_4;
        case 4:
                model_stages[3] = &stage_3;
        case 3:
                model_stages[2] = &stage_2;
        case 2:
                model_stages[1] = &stage_1;
        case 1:
                model_stages[0] = &stage_0;
        case 0:
        break;
    }

    face_cascade->stages = model_stages;

    unsigned max_cascade_size = biggest_cascade_stage(face_cascade);
    PRINTF("Max cascade size:%u\n", max_cascade_size);

    for(int i = 0; i < CASCADE_STAGES_L1; i++)
        face_cascade->stages[i] = sync_copy_cascade_stage_to_l1(cl, (face_cascade->stages[i]));

    if (CASCADE_STAGES_L1 < CASCADE_TOTAL_STAGES)
    {
        face_cascade->buffers_l1[0] = pi_l1_malloc(cl, max_cascade_size);
        if (face_cascade->buffers_l1[0] == NULL)
        {
            PRINTF("Error: Failed to allocate cascade buffers\n");
            return NULL;
        }
    }
    if (CASCADE_STAGES_L1 < CASCADE_TOTAL_STAGES - 1)
    {
        face_cascade->buffers_l1[1] = pi_l1_malloc(cl, max_cascade_size);
        if (face_cascade->buffers_l1[1] == NULL)
        {
            PRINTF("Error: Failed to allocate cascade buffers\n");
            return NULL;
        }
    }

    return face_cascade;
}

static unsigned biggest_cascade_stage(const cascade_t *cascade)
{
    //Calculate cascade bigger layer
    unsigned max_stage_size = 0;

    for (int i = 0; i < CASCADE_TOTAL_STAGES; i++)
    {
        single_cascade_t *stage = cascade->stages[i];
        unsigned stage_size;

        stage_size = sizeof(*stage) +
                     stage->stage_size * (
                         sizeof(*stage->thresholds) +
                         sizeof(*stage->alpha1) +
                         sizeof(*stage->alpha2) +
                         sizeof(*stage->rect_num)
                     ) + sizeof(*stage->rect_num) +
                     stage->rectangles_size * sizeof(*stage->rectangles) +
                     (stage->rectangles_size/4) * sizeof(*stage->weights);

        if (stage_size > max_stage_size)
            max_stage_size = stage_size;
        //PRINTF ("Stage size: %u\n", stage_size);
    }

    return max_stage_size;
}

static int rect_intersect_area(
        unsigned short a_x, unsigned short a_y, unsigned short a_w, unsigned short a_h,
        unsigned short b_x, unsigned short b_y, unsigned short b_w, unsigned short b_h)
{
    #define MIN(a,b) ((a) < (b) ? (a) : (b))
    #define MAX(a,b) ((a) > (b) ? (a) : (b))

    int x = MAX(a_x,b_x);
    int y = MAX(a_y,b_y);

    int size_x = MIN(a_x+a_w,b_x+b_w) - x;
    int size_y = MIN(a_y+a_h,b_y+b_h) - y;

    if(size_x <= 0 || size_y <= 0)
        return 0;
    else
        return size_x*size_y;

    #undef MAX
    #undef MIN
}

static void non_max_suppress(cascade_reponse_t* responses, int response_idx)
{
    int idx;

    //Non-max supression
    for(idx = 0; idx < response_idx; idx++){
        //check if rect has been removed (-1)
        if(responses[idx].x == -1)
            continue;

        int idx_int;
        for(idx_int = idx + 1; idx_int < response_idx; idx_int++){

            if(responses[idx_int].x == -1)
                continue;

            //check the intersection between rects
            int intersection = rect_intersect_area(responses[idx].x,responses[idx].y,responses[idx].w,responses[idx].h,
                                                   responses[idx_int].x,responses[idx_int].y,responses[idx_int].w,responses[idx_int].h);

            if(intersection >= NON_MAX_THRES)
            {   //is non-max
                //supress the one that has lower score
                if(responses[idx_int].score > responses[idx].score){
                    responses[idx].x = -1;
                    responses[idx].y = -1;
                }else{
                    responses[idx_int].x = -1;
                    responses[idx_int].y = -1;
                }
            }
        }
    }
}

void cascade_detect(ArgCluster_T *ArgC)
{
    unsigned int Wout = WOUT_INIT, Hout = HOUT_INIT;
    unsigned int Win = ArgC->Win, Hin = ArgC->Hin;
    int response_idx = 0;
    int result;

    //create structure for output
    cascade_reponse_t* responses = ArgC->reponses;
    for(int i = 0; i < MAX_NUM_OUT_WINS; i++)
        responses[i].x=-1;

#ifdef ENABLE_LAYER_1
    ResizeImage_1(ArgC->ImageIn,ArgC->ImageOut);
    ProcessIntegralImage_1(ArgC->ImageOut,ArgC->ImageIntegral);
    ProcessSquaredIntegralImage_1(ArgC->ImageOut,ArgC->SquaredImageIntegral);
    ProcessCascade_1(ArgC->ImageIntegral, ArgC->SquaredImageIntegral, ArgC->model, ArgC->output_map);

    for(unsigned int i=0;i<Hout-24+1;i+=DETECT_STRIDE)
        for(unsigned int j=0;j<Wout-24+1;j+=DETECT_STRIDE){
            result = ArgC->output_map[i*(Wout-24+1)+j];

            if(result!=0){
                responses[response_idx].x = (j*Win)/Wout;
                responses[response_idx].y = (i*Hin)/Hout;
                responses[response_idx].w = ((j+24)*Win+Wout-1)/Wout - responses[response_idx].x + 1;
                responses[response_idx].h = ((i+24)*Hin+Hout-1)/Hout - responses[response_idx].y + 1;
                responses[response_idx].score  = result;
                responses[response_idx].layer_idx = 0;
                response_idx++;
                // PRINTF("Face Found on layer 1 in %dx%d at X: %d, Y: %d - value: %d\n",Wout,Hout,j,i,result);
            }
        }
#endif

    Wout /= 1.25, Hout /= 1.25;

#ifdef ENABLE_LAYER_2
    ResizeImage_2(ArgC->ImageIn,ArgC->ImageOut);
    ProcessIntegralImage_2(ArgC->ImageOut,ArgC->ImageIntegral);
    ProcessSquaredIntegralImage_2(ArgC->ImageOut,ArgC->SquaredImageIntegral);
    ProcessCascade_2(ArgC->ImageIntegral,ArgC->SquaredImageIntegral,ArgC->model, ArgC->output_map);

    for(unsigned int i=0;i<Hout-24+1;i+=DETECT_STRIDE)
        for(unsigned int j=0;j<Wout-24+1;j+=DETECT_STRIDE){

            result = ArgC->output_map[i*(Wout-24+1)+j];
            if(result!=0){
                responses[response_idx].x = (j*Win)/Wout;
                responses[response_idx].y = (i*Hin)/Hout;
                responses[response_idx].w = ((j+24)*Win+Wout-1)/Wout - responses[response_idx].x + 1;
                responses[response_idx].h = ((i+24)*Hin+Hout-1)/Hout - responses[response_idx].y + 1;
                responses[response_idx].score = result;
                responses[response_idx].layer_idx = 1;
                response_idx++;
                // PRINTF("Face Found on layer 2 in %dx%d at X: %d, Y: %d - value: %d\n",Wout,Hout,j,i,result);
            }
        }
#endif

    Wout /= 1.25, Hout /= 1.25;

#ifdef ENABLE_LAYER_3
    ResizeImage_3(ArgC->ImageIn,ArgC->ImageOut);
    ProcessIntegralImage_3(ArgC->ImageOut,ArgC->ImageIntegral);
    ProcessSquaredIntegralImage_3(ArgC->ImageOut,ArgC->SquaredImageIntegral);
    ProcessCascade_3(ArgC->ImageIntegral,ArgC->SquaredImageIntegral,ArgC->model, ArgC->output_map);

    for(unsigned int i=0;i<Hout-24+1;i+=DETECT_STRIDE)
        for(unsigned int j=0;j<Wout-24+1;j+=DETECT_STRIDE){

            result = ArgC->output_map[i*(Wout-24+1)+j];
            if(result!=0){
                responses[response_idx].x = (j*Win)/Wout;
                responses[response_idx].y = (i*Hin)/Hout;
                responses[response_idx].w = ((j+24)*Win+Wout-1)/Wout - responses[response_idx].x + 1;
                responses[response_idx].h = ((i+24)*Hin+Hout-1)/Hout - responses[response_idx].y + 1;
                responses[response_idx].score = result;
                responses[response_idx].layer_idx = 2;
                response_idx++;
                // PRINTF("Face Found on layer 3 in %dx%d at X: %d, Y: %d - value: %d\n",Wout,Hout,j,i,result);
            }
        }
#endif

    non_max_suppress(responses, response_idx);

    ArgC->num_reponse = response_idx;
}
