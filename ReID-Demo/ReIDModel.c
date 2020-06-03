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

#include <stdint.h>
#include <stdio.h>
#include <string.h>
// Import AutoTiler lib
#include "AutoTilerLib.h"
// Import CNN generators
#include "CNN_Generators.h"
#include "param_layer_struct.h"

void CnnModel(unsigned int L1Memory, unsigned int L2Memory)
{
    SetInlineMode(ALWAYS_INLINE);
    SetSymbolDynamics();
#if defined(_FOR_FREERTOS_)
    SetUsedFilesNames(0, 2, "pmsis_tiling.h", "CNN_BasicKernels.h");
#else
    SetUsedFilesNames(0, 1, "CNN_BasicKernels.h");
#endif
    SetGeneratedFilesNames("CnnKernels.c", "CnnKernels.h");

    int L3Memory=8*1024*1024;
    SetMemoryDeviceInfos(3,
        AT_MEM_L1, L1Memory, "L1_Memory", 0, 0,
        AT_MEM_L2, L2Memory, "L2_Memory", 0, 0,
        AT_MEM_L3_HRAM, L3Memory, "L3_Memory", 0, 1
    );


    LoadCNNLibrary();
    //CNN_LoadHWCEKernelLibrary();

    for (size_t i = 0; i < NB_CONV; i++)
    {
        CNN_ConvolutionPoolReLU(
            convLayers[i].name,
            NULL,
            2,2,2,2,             // All short ints
            convLayers[i].q.in,  // Input quantization
            convLayers[i].q.weights, // Weight quantization
            convLayers[i].q.bias,// Bias quantization
            convLayers[i].q.out, // Output quantization
            0,1,1,0,
            convLayers[i].nb_if, // InFeat
            convLayers[i].nb_of, // OutFeat
            convLayers[i].win,   // Width
            convLayers[i].hin,   // Height
            KOP_CONV,
            convLayers[i].kernel_width, // FScW
            convLayers[i].kernel_height, // FScH
            1,
            1,
            convLayers[i].conv_stride, // ConvStrideW
            convLayers[i].conv_stride, // ConvStrideH
            convLayers[i].conv_padding,

            convLayers[i].max_pool?KOP_MAXPOOL:KOP_NONE, //Max Pool convLayers[i].max_pool
            convLayers[i].pool_size, // FSpW
            convLayers[i].pool_size, // FSpH
            1,  //Dilation x
            1,  //Dilation y
            convLayers[i].pool_stride, // PoolStrideW
            convLayers[i].pool_stride, // PoolStrideH
            0, // PoolDoPad

            convLayers[i].relu?KOP_RELU:KOP_NONE      //     convLayers[i].relu
        );
    }


    CNN_PoolReLU(
        "GPool10", // Name
        NULL,
        2, // In_DataSize
        2, // Out_DataSize
        convLayers[NB_CONV-1].q.out,
        convLayers[NB_CONV-1].q.out,
        0, // In_InL3
        0, // Out_InL3

        512, // InFeat
        512, // OutFeat
        7,   // Width
        7,   // Height

        KOP_AVGPOOL, //AVERAGE POOLING
        7,   // FSpW
        7,   // FSpH
        1,   //Dilation
        1,   //Dilation
        1,   // PoolStrideW
        1,   // PoolStrideH
        0,   // PoolDoPad
        KOP_NONE
    );
}

int main(int argc, char **argv)
{
    if (TilerParseOptions(argc, argv))
    {
        printf("Failed to initialize or incorrect output directory.\n");
        return -1;
    }

    CnnModel(45000, 50000);
    GenerateTilingCode();
    return 0;
}
