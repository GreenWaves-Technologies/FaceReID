#ifndef __FACEDET_PIPELINE_H__
#define __FACEDET_PIPELINE_H__

#include "cascade.h"

int check_detection_stability(cascade_reponse_t* hisotry, int history_size);
void detection_cluster_init(ArgCluster_T *ArgC);
void detection_cluster_main(ArgCluster_T *ArgC);

#endif