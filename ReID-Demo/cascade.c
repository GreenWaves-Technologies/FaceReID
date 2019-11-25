#include "pmsis.h"

#if defined(__FREERTOS__)
# include "dma/cl_dma.h"
# include "pmsis_os.h"
# include "pmsis_l1_malloc.h"
# include "pmsis_tiling.h"
#else
# include "Gap.h"
# include "extra_emul_stubs.h"
#endif

#include "cascade.h"
#include "setup.h"
#include "face_cascade.h"
#include "FaceDetKernels.h"

#include <stdlib.h>
#include <stdio.h>

//Permanently Store a scascade stage to L1
single_cascade_t* sync_copy_cascade_stage_to_l1(single_cascade_t* cascade_l2)
{
    pi_cl_dma_copy_t DmaR_Evt1;

    single_cascade_t* cascade_l1;
    cascade_l1 = (single_cascade_t* )pmsis_l1_malloc( sizeof(single_cascade_t));

    cascade_l1->stage_size = cascade_l2->stage_size;
    cascade_l1->rectangles_size = cascade_l2->rectangles_size;

    cascade_l1->thresholds     = (short*)pmsis_l1_malloc( sizeof(short)*cascade_l2->stage_size);
    pi_cl_dma_cmd((unsigned int) cascade_l2->thresholds, (unsigned int) cascade_l1->thresholds, sizeof(short)*cascade_l1->stage_size, PI_CL_DMA_DIR_EXT2LOC, &DmaR_Evt1);
    pi_cl_dma_cmd_wait(&DmaR_Evt1);

    cascade_l1->alpha1         = (short*)pmsis_l1_malloc( sizeof(short)*cascade_l2->stage_size);
    pi_cl_dma_cmd((unsigned int) cascade_l2->alpha1, (unsigned int) cascade_l1->alpha1, sizeof(short)*cascade_l1->stage_size, PI_CL_DMA_DIR_EXT2LOC, &DmaR_Evt1);
    pi_cl_dma_cmd_wait(&DmaR_Evt1);

    cascade_l1->alpha2         = (short*)pmsis_l1_malloc( sizeof(short)*cascade_l2->stage_size);
    pi_cl_dma_cmd((unsigned int) cascade_l2->alpha2, (unsigned int) cascade_l1->alpha2, sizeof(short)*cascade_l1->stage_size, PI_CL_DMA_DIR_EXT2LOC, &DmaR_Evt1);
    pi_cl_dma_cmd_wait(&DmaR_Evt1);

    cascade_l1->rect_num       = (unsigned  short*)pmsis_l1_malloc( sizeof(unsigned short)*((cascade_l2->stage_size)+1));
    pi_cl_dma_cmd((unsigned int) cascade_l2->rect_num, (unsigned int) cascade_l1->rect_num, sizeof(unsigned short)*(cascade_l1->stage_size+1), PI_CL_DMA_DIR_EXT2LOC, &DmaR_Evt1);
    pi_cl_dma_cmd_wait(&DmaR_Evt1);

    cascade_l1->weights    = (signed char*)pmsis_l1_malloc( sizeof(signed char)*(cascade_l2->rectangles_size/4));
    pi_cl_dma_cmd((unsigned int) cascade_l2->weights, (unsigned int) cascade_l1->weights, sizeof(signed char)*(cascade_l2->rectangles_size/4), PI_CL_DMA_DIR_EXT2LOC, &DmaR_Evt1);
    pi_cl_dma_cmd_wait(&DmaR_Evt1);

    cascade_l1->rectangles = (char*)pmsis_l1_malloc( sizeof(char)*cascade_l2->rectangles_size);
    pi_cl_dma_cmd((unsigned int) cascade_l2->rectangles, (unsigned int) cascade_l1->rectangles, sizeof(char)*cascade_l2->rectangles_size, PI_CL_DMA_DIR_EXT2LOC, &DmaR_Evt1);
    pi_cl_dma_cmd_wait(&DmaR_Evt1);

    if(cascade_l1->rectangles==0)
        PRINTF("Allocation Error...\n");

    return cascade_l1;
}

cascade_t *getFaceCascade(){
    cascade_t *face_cascade;

    face_cascade = (cascade_t*) pmsis_l1_malloc( sizeof(cascade_t));
    if(face_cascade==0){
        PRINTF("Error allocatin model thresholds...");
        return 0;
    }
    single_cascade_t **model_stages = (single_cascade_t**) pmsis_l1_malloc( sizeof(single_cascade_t*)*CASCADE_TOTAL_STAGES);

    face_cascade->stages_num = CASCADE_TOTAL_STAGES;
    face_cascade->thresholds = (signed short *) pmsis_l1_malloc( sizeof(signed short )*face_cascade->stages_num);
    if(face_cascade->thresholds==0){
        PRINTF("Error allocatin model thresholds...");
        return 0;
    }

    for(int a=0; a<face_cascade->stages_num; a++)
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

    int max_cascade_size = biggest_cascade_stage(face_cascade);
    PRINTF("Max cascade size:%d\n",max_cascade_size);

    for(int i=0; i<CASCADE_STAGES_L1; i++)
        face_cascade->stages[i] = sync_copy_cascade_stage_to_l1((face_cascade->stages[i]));

    face_cascade->buffers_l1[0] = pmsis_l1_malloc(max_cascade_size);
    face_cascade->buffers_l1[1] = pmsis_l1_malloc(max_cascade_size);

    if(face_cascade->buffers_l1[0]==0 ){
        PRINTF("Error allocating cascade buffer 0...\n");
    }

    if(face_cascade->buffers_l1[1] == 0){
        PRINTF("Error allocating cascade buffer 1...\n");
    }


    return face_cascade;
}

int biggest_cascade_stage(cascade_t *cascade){

    //Calculate cascade bigger layer
    int biggest_stage_size=0;
    int cur_layer;

    for (int i=0; i<cascade->stages_num; i++) {

        cur_layer = sizeof(cascade->stages[i]->stage_size) +
                           sizeof(cascade->stages[i]->rectangles_size) +
                                (cascade->stages[i]->stage_size*
                                        (sizeof(cascade->stages[i]->thresholds) +
                                            sizeof(cascade->stages[i]->alpha1) +
                                            sizeof(cascade->stages[i]->alpha2) +
                                            sizeof(cascade->stages[i]->rect_num)
                                        )
                                ) +
                                (cascade->stages[i]->rectangles_size*sizeof(cascade->stages[i]->rectangles)) +
                                ((cascade->stages[i]->rectangles_size/4)*sizeof(cascade->stages[i]->weights));

        if(cur_layer>biggest_stage_size)
                biggest_stage_size=cur_layer;
        //PRINTF ("Stage size: %d\n",cur_layer);
    }

    return biggest_stage_size;
}

int rect_intersect_area(unsigned short a_x, unsigned short a_y, unsigned short a_w, unsigned short a_h,
                        unsigned short b_x, unsigned short b_y, unsigned short b_w, unsigned short b_h ){

    #define MIN(a,b) ((a) < (b) ? (a) : (b))
    #define MAX(a,b) ((a) > (b) ? (a) : (b))

    int x = MAX(a_x,b_x);
    int y = MAX(a_y,b_y);

    int size_x = MIN(a_x+a_w,b_x+b_w) - x;
    int size_y = MIN(a_y+a_h,b_y+b_h) - y;

    if(size_x <=0 || size_y <=0)
        return 0;
    else
        return size_x*size_y;

    #undef MAX
    #undef MIN
}

void non_max_suppress(cascade_reponse_t* reponses, int reponse_idx){

    int idx,idx_int;

    //Non-max supression
    for(idx=0;idx<reponse_idx;idx++){
        //check if rect has been removed (-1)
        if(reponses[idx].x==-1)
            continue;

        for(idx_int=0;idx_int<reponse_idx;idx_int++){

            if(reponses[idx_int].x==-1 || idx_int==idx)
                continue;

            //check the intersection between rects
            int intersection = rect_intersect_area(reponses[idx].x,reponses[idx].y,reponses[idx].w,reponses[idx].h,
                                                   reponses[idx_int].x,reponses[idx_int].y,reponses[idx_int].w,reponses[idx_int].h);

            if(intersection >= NON_MAX_THRES)
            {   //is non-max
                //supress the one that has lower score
                if(reponses[idx_int].score > reponses[idx].score){
                    reponses[idx].x = -1;
                    reponses[idx].y = -1;
                }else{
                    reponses[idx_int].x = -1;
                    reponses[idx_int].y = -1;
                }
            }
        }
    }
}

void cascade_detect(ArgCluster_T *ArgC)
{
    unsigned int Wout = WOUT_INIT, Hout = HOUT_INIT;
    unsigned int Win=ArgC->Win, Hin=ArgC->Hin;
    int reponse_idx = 0;
    int result;

    //create structure for output
    cascade_reponse_t* reponses = ArgC->reponses;
    for(int i = 0; i < MAX_NUM_OUT_WINS; i++)
        reponses[i].x=-1;

#ifdef ENABLE_LAYER_1
    ResizeImage_1(ArgC->ImageIn,ArgC->ImageOut);
    ProcessIntegralImage_1(ArgC->ImageOut,ArgC->ImageIntegral);
    ProcessSquaredIntegralImage_1(ArgC->ImageOut,ArgC->SquaredImageIntegral);
    ProcessCascade_1(ArgC->ImageIntegral, ArgC->SquaredImageIntegral, ArgC->model, ArgC->output_map);

    for(unsigned int i=0;i<Hout-24+1;i+=DETECT_STRIDE)
        for(unsigned int j=0;j<Wout-24+1;j+=DETECT_STRIDE){
            result = ArgC->output_map[i*(Wout-24+1)+j];

            if(result!=0){
                reponses[reponse_idx].x = (j*Win)/Wout;
                reponses[reponse_idx].y = (i*Hin)/Hout;
                reponses[reponse_idx].w = (24*Win)/Wout;
                reponses[reponse_idx].h = (24*Hin)/Hout;
                reponses[reponse_idx].score   = result;
                reponses[reponse_idx].layer_idx = 0;
                reponse_idx++;
                PRINTF("Face Found on layer 1 in %dx%d at X: %d, Y: %d - value: %d\n",Wout,Hout,j,i,result);
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
                reponses[reponse_idx].x = (j*Win)/Wout;
                reponses[reponse_idx].y = (i*Hin)/Hout;
                reponses[reponse_idx].w = (24*Win)/Wout;
                reponses[reponse_idx].h = (24*Hin)/Hout;
                reponses[reponse_idx].score = result;
                reponses[reponse_idx].layer_idx = 1;
                reponse_idx++;
                PRINTF("Face Found on layer 2 in %dx%d at X: %d, Y: %d - value: %d\n",Wout,Hout,j,i,result);
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
                reponses[reponse_idx].x = (j*Win)/Wout;
                reponses[reponse_idx].y = (i*Hin)/Hout;
                reponses[reponse_idx].w = (24*Win)/Wout;
                reponses[reponse_idx].h = (24*Hin)/Hout;
                reponses[reponse_idx].score = result;
                reponses[reponse_idx].layer_idx = 2;
                reponse_idx++;
                PRINTF("Face Found on layer 3 in %dx%d at X: %d, Y: %d - value: %d\n",Wout,Hout,j,i,result);
            }
        }
#endif

    non_max_suppress(reponses,reponse_idx);

    ArgC->num_reponse=reponse_idx;
}
