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
#include "FireGenerator.h"
#include "CNN_Generators.h"
#include "param_layer_struct.h"
#include "setup.h"

void CnnModel(unsigned int L1Memory, unsigned int L2Memory, unsigned int L3Memory, unsigned int L3Flash)
{
    CNN_GenControl_T Ctrl;
    CNN_InitGenCtrl(&Ctrl);

    SetInlineMode(ALWAYS_INLINE);
    SetSymbolDynamics();

    SetUsedFilesNames(0, 1, "CNN_BasicKernels.h");
    SetGeneratedFilesNames("CnnKernels.c", "CnnKernels.h");

    SetMemoryDeviceInfos(4,
        AT_MEM_L1, L1Memory, "L1_Memory", 0, 1,
        AT_MEM_L2, L2Memory, "L2_Memory", 0, 1,
        AT_MEM_L3_HRAM, L3Memory, "L3_Memory", 0, 1,
        AT_MEM_L3_HFLASH, L3Flash, "0", "SqueezeNet_L3_Flash_Const.dat", 0
    );

    CNN_SetGenCtrl(&Ctrl, "EnableIm2Col", AT_OPT_ON);
    AT_SetGraphCtrl(AT_GRAPH_PRODUCE_NODE_NAMES, AT_OPT_ON);
    AT_SetGraphCtrl(AT_GRAPH_PRODUCE_NODE_CVAR_NAME, "SqueezeNetLName");
    AT_SetGraphCtrl(AT_GRAPH_NOINLINE_NODE, AT_OPT_ON);
    //AT_SetGraphCtrl(AT_GRAPH_TRACE_EXEC, AT_OPT_ON);
#ifdef PERF_COUNT
    AT_SetGraphCtrl(AT_GRAPH_MONITOR_CYCLES, AT_OPT_ON);
    AT_SetGraphCtrl(AT_GRAPH_MONITOR_CVAR_NAME, "SqueezeNetPerf");
    AT_SetGraphCtrl(AT_GRAPH_PRODUCE_OPERINFOS, AT_OPT_ON);
    AT_SetGraphCtrl(AT_GRAPH_PRODUCE_OPERINFOS_CVAR_NAME, "SqueezeNetOperCount");
#endif
    AT_SetGraphCtrl(AT_GRAPH_REORDER_CONSTANT_IN, AT_OPT_ON);
    AT_SetGraphCtrl(AT_GRAPH_CONST_EXEC_FROM_FLASH, AT_OPT_ON);
    SetAT_TestFile("AT_SqueezeNet.inc");

    LoadCNNLibrary();

#ifdef GRAPH
    int w_InL3 = 0, b_InL3 = 0;
#else
    int w_InL3 = 1, b_InL3 = 1;
#endif

    // Conv0MP0, Conv1MP1
    for (unsigned i = 0; i < 2; i++)
    {
        CNN_ConvolutionPoolReLU(
            convLayers[i].name,
            &Ctrl,
            2,2,2,2,             // All short ints
            convLayers[i].q.in,  // Input quantization
            convLayers[i].q.weights, // Weight quantization
            convLayers[i].q.bias,// Bias quantization
            convLayers[i].q.out, // Output quantization
            0, w_InL3, b_InL3, 0,
            convLayers[i].nb_if, // InFeat
            convLayers[i].nb_of, // OutFeat
            convLayers[i].win,   // Width
            convLayers[i].hin,   // Height
            KOP_CONV,
            convLayers[i].kernel_width,  // FScW
            convLayers[i].kernel_height, // FScH
            1,
            1,
            convLayers[i].conv_stride, // ConvStrideW
            convLayers[i].conv_stride, // ConvStrideH
            convLayers[i].conv_padding,

            convLayers[i].max_pool ? KOP_MAXPOOL : KOP_NONE, // Max Pool
            convLayers[i].pool_size, // FSpW
            convLayers[i].pool_size, // FSpH
            1,  //Dilation x
            1,  //Dilation y
            convLayers[i].pool_stride, // PoolStrideW
            convLayers[i].pool_stride, // PoolStrideH
            0, // PoolDoPad

            convLayers[i].relu ? KOP_RELU : KOP_NONE
        );
    }

    Fire("Fire3", &Ctrl, 2);
    Fire("Fire4", &Ctrl, 5);
    Fire("Fire6", &Ctrl, 8);
    Fire("Fire7", &Ctrl, 11);
    Fire("Fire9", &Ctrl, 14);
    Fire("Fire10", &Ctrl, 17);
    Fire("Fire11", &Ctrl, 20);
    Fire("Fire12", &Ctrl, 23);

    CNN_PoolReLU(
        "GPool10", // Name
        &Ctrl,
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

#ifdef GRAPH
    CreateGraph("SqueezeNetCNN",
            /* Arguments either passed or globals */
        CArgs(1 + 2 + 2 + 8*6 + 1,

            TCArgInfoA("short * __restrict__", "NetworkIn", ARG_SCOPE_ARG, ARG_DIR_IN, AT_MEM_L2, AT_MEM_UNDEF, NULL),

            TCArgInfo ("short * __restrict__", "Conv0MP0_W", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/conv1.0.weights.bin", 1,1, 16, convLayers[0].q.weights)),
            TCArgInfo ("short * __restrict__", "Conv0MP0_B", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/conv1.0.bias.bin", 1,1, 16, convLayers[0].q.bias)),

            TCArgInfo ("short * __restrict__", "Conv1MP1_W", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.0.0.weights.bin", 1,1, 16, convLayers[1].q.weights)),
            TCArgInfo ("short * __restrict__", "Conv1MP1_B", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.0.0.bias.bin", 1,1, 16, convLayers[1].q.bias)),

            TCArgInfo ("short * __restrict__", "Fire3_W0", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.3.squeeze.0.weights.bin", 1,1, 16, convLayers[2].q.weights)),
            TCArgInfo ("short * __restrict__", "Fire3_B0", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.3.squeeze.0.bias.bin", 1,1, 16, convLayers[2].q.bias)),
            TCArgInfo ("short * __restrict__", "Fire3_W1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.3.expand1x1.0.weights.bin", 1,1, 16, convLayers[3].q.weights)),
            TCArgInfo ("short * __restrict__", "Fire3_B1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.3.expand1x1.0.bias.bin", 1,1, 16, convLayers[3].q.bias)),
            TCArgInfo ("short * __restrict__", "Fire3_W2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.3.expand3x3.0.weights.bin", 1,1, 16, convLayers[4].q.weights)),
            TCArgInfo ("short * __restrict__", "Fire3_B2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.3.expand3x3.0.bias.bin", 1,1, 16, convLayers[4].q.bias)),

            TCArgInfo ("short * __restrict__", "Fire4_W0", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.4.squeeze.0.weights.bin", 1,1, 16, convLayers[5].q.weights)),
            TCArgInfo ("short * __restrict__", "Fire4_B0", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.4.squeeze.0.bias.bin", 1,1, 16, convLayers[5].q.bias)),
            TCArgInfo ("short * __restrict__", "Fire4_W1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.4.expand1x1.0.weights.bin", 1,1, 16, convLayers[6].q.weights)),
            TCArgInfo ("short * __restrict__", "Fire4_B1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.4.expand1x1.0.bias.bin", 1,1, 16, convLayers[6].q.bias)),
            TCArgInfo ("short * __restrict__", "Fire4_W2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.4.expand3x3.0.weights.bin", 1,1, 16, convLayers[7].q.weights)),
            TCArgInfo ("short * __restrict__", "Fire4_B2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.4.expand3x3.0.bias.bin", 1,1, 16, convLayers[7].q.bias)),

            TCArgInfo ("short * __restrict__", "Fire6_W0", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.6.squeeze.0.weights.bin", 1,1, 16, convLayers[8].q.weights)),
            TCArgInfo ("short * __restrict__", "Fire6_B0", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.6.squeeze.0.bias.bin", 1,1, 16, convLayers[8].q.bias)),
            TCArgInfo ("short * __restrict__", "Fire6_W1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.6.expand1x1.0.weights.bin", 1,1, 16, convLayers[9].q.weights)),
            TCArgInfo ("short * __restrict__", "Fire6_B1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.6.expand1x1.0.bias.bin", 1,1, 16, convLayers[9].q.bias)),
            TCArgInfo ("short * __restrict__", "Fire6_W2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.6.expand3x3.0.weights.bin", 1,1, 16, convLayers[10].q.weights)),
            TCArgInfo ("short * __restrict__", "Fire6_B2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.6.expand3x3.0.bias.bin", 1,1, 16, convLayers[10].q.bias)),

            TCArgInfo ("short * __restrict__", "Fire7_W0", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.7.squeeze.0.weights.bin", 1,1, 16, convLayers[11].q.weights)),
            TCArgInfo ("short * __restrict__", "Fire7_B0", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.7.squeeze.0.bias.bin", 1,1, 16, convLayers[11].q.bias)),
            TCArgInfo ("short * __restrict__", "Fire7_W1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.7.expand1x1.0.weights.bin", 1,1, 16, convLayers[12].q.weights)),
            TCArgInfo ("short * __restrict__", "Fire7_B1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.7.expand1x1.0.bias.bin", 1,1, 16, convLayers[12].q.bias)),
            TCArgInfo ("short * __restrict__", "Fire7_W2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.7.expand3x3.0.weights.bin", 1,1, 16, convLayers[13].q.weights)),
            TCArgInfo ("short * __restrict__", "Fire7_B2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.7.expand3x3.0.bias.bin", 1,1, 16, convLayers[13].q.bias)),

            TCArgInfo ("short * __restrict__", "Fire9_W0", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.9.squeeze.0.weights.bin", 1,1, 16, convLayers[14].q.weights)),
            TCArgInfo ("short * __restrict__", "Fire9_B0", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.9.squeeze.0.bias.bin", 1,1, 16, convLayers[14].q.bias)),
            TCArgInfo ("short * __restrict__", "Fire9_W1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.9.expand1x1.0.weights.bin", 1,1, 16, convLayers[15].q.weights)),
            TCArgInfo ("short * __restrict__", "Fire9_B1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.9.expand1x1.0.bias.bin", 1,1, 16, convLayers[15].q.bias)),
            TCArgInfo ("short * __restrict__", "Fire9_W2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.9.expand3x3.0.weights.bin", 1,1, 16, convLayers[16].q.weights)),
            TCArgInfo ("short * __restrict__", "Fire9_B2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.9.expand3x3.0.bias.bin", 1,1, 16, convLayers[16].q.bias)),

            TCArgInfo ("short * __restrict__", "Fire10_W0", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.10.squeeze.0.weights.bin", 1,1, 16, convLayers[17].q.weights)),
            TCArgInfo ("short * __restrict__", "Fire10_B0", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.10.squeeze.0.bias.bin", 1,1, 16, convLayers[17].q.bias)),
            TCArgInfo ("short * __restrict__", "Fire10_W1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.10.expand1x1.0.weights.bin", 1,1, 16, convLayers[18].q.weights)),
            TCArgInfo ("short * __restrict__", "Fire10_B1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.10.expand1x1.0.bias.bin", 1,1, 16, convLayers[18].q.bias)),
            TCArgInfo ("short * __restrict__", "Fire10_W2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.10.expand3x3.0.weights.bin", 1,1, 16, convLayers[19].q.weights)),
            TCArgInfo ("short * __restrict__", "Fire10_B2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.10.expand3x3.0.bias.bin", 1,1, 16, convLayers[19].q.bias)),

            TCArgInfo ("short * __restrict__", "Fire11_W0", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.11.squeeze.0.weights.bin", 1,1, 16, convLayers[20].q.weights)),
            TCArgInfo ("short * __restrict__", "Fire11_B0", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.11.squeeze.0.bias.bin", 1,1, 16, convLayers[20].q.bias)),
            TCArgInfo ("short * __restrict__", "Fire11_W1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.11.expand1x1.0.weights.bin", 1,1, 16, convLayers[21].q.weights)),
            TCArgInfo ("short * __restrict__", "Fire11_B1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.11.expand1x1.0.bias.bin", 1,1, 16, convLayers[21].q.bias)),
            TCArgInfo ("short * __restrict__", "Fire11_W2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.11.expand3x3.0.weights.bin", 1,1, 16, convLayers[22].q.weights)),
            TCArgInfo ("short * __restrict__", "Fire11_B2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.11.expand3x3.0.bias.bin", 1,1, 16, convLayers[22].q.bias)),

            TCArgInfo ("short * __restrict__", "Fire12_W0", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.12.squeeze.0.weights.bin", 1,1, 16, convLayers[23].q.weights)),
            TCArgInfo ("short * __restrict__", "Fire12_B0", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.12.squeeze.0.bias.bin", 1,1, 16, convLayers[23].q.bias)),
            TCArgInfo ("short * __restrict__", "Fire12_W1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.12.expand1x1.0.weights.bin", 1,1, 16, convLayers[24].q.weights)),
            TCArgInfo ("short * __restrict__", "Fire12_B1", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.12.expand1x1.0.bias.bin", 1,1, 16, convLayers[24].q.bias)),
            TCArgInfo ("short * __restrict__", "Fire12_W2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.12.expand3x3.0.weights.bin", 1,1, 16, convLayers[25].q.weights)),
            TCArgInfo ("short * __restrict__", "Fire12_B2", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo(CNN_LAYERS_PATH "/features.12.expand3x3.0.bias.bin", 1,1, 16, convLayers[25].q.bias)),

            TCArgInfoA("short * __restrict__", "NetworkOut",   ARG_SCOPE_ARG,     ARG_DIR_OUT, AT_MEM_UNDEF, AT_MEM_L2, NULL)
        ),
        /* Locals */
        CArgs(1 + 1 + 8*2,

            TCArgInfo ("short * __restrict__", "Conv0MP0_Out", ARG_SCOPE_LOCAL,  ARG_DIR_INOUT,  AT_MEM_UNDEF, AT_MEM_UNDEF, NULL),

            TCArgInfo ("short * __restrict__", "Conv1MP1_Out", ARG_SCOPE_LOCAL,  ARG_DIR_INOUT,  AT_MEM_UNDEF, AT_MEM_UNDEF, NULL),

            TCArgInfo ("short * __restrict__", "Fire3_O0",     ARG_SCOPE_LOCAL,  ARG_DIR_INOUT,  AT_MEM_UNDEF, AT_MEM_UNDEF, NULL),
            TCArgInfo ("short * __restrict__", "Fire3_Out",    ARG_SCOPE_LOCAL,  ARG_DIR_INOUT,  AT_MEM_UNDEF, AT_MEM_UNDEF, NULL),

            TCArgInfo ("short * __restrict__", "Fire4_O0",     ARG_SCOPE_LOCAL,  ARG_DIR_INOUT,  AT_MEM_UNDEF, AT_MEM_UNDEF, NULL),
            TCArgInfo ("short * __restrict__", "Fire4_Out",    ARG_SCOPE_LOCAL,  ARG_DIR_INOUT,  AT_MEM_UNDEF, AT_MEM_UNDEF, NULL),

            TCArgInfo ("short * __restrict__", "Fire6_O0",     ARG_SCOPE_LOCAL,  ARG_DIR_INOUT,  AT_MEM_UNDEF, AT_MEM_UNDEF, NULL),
            TCArgInfo ("short * __restrict__", "Fire6_Out",    ARG_SCOPE_LOCAL,  ARG_DIR_INOUT,  AT_MEM_UNDEF, AT_MEM_UNDEF, NULL),

            TCArgInfo ("short * __restrict__", "Fire7_O0",     ARG_SCOPE_LOCAL,  ARG_DIR_INOUT,  AT_MEM_UNDEF, AT_MEM_UNDEF, NULL),
            TCArgInfo ("short * __restrict__", "Fire7_Out",    ARG_SCOPE_LOCAL,  ARG_DIR_INOUT,  AT_MEM_UNDEF, AT_MEM_UNDEF, NULL),

            TCArgInfo ("short * __restrict__", "Fire9_O0",     ARG_SCOPE_LOCAL,  ARG_DIR_INOUT,  AT_MEM_UNDEF, AT_MEM_UNDEF, NULL),
            TCArgInfo ("short * __restrict__", "Fire9_Out",    ARG_SCOPE_LOCAL,  ARG_DIR_INOUT,  AT_MEM_UNDEF, AT_MEM_UNDEF, NULL),

            TCArgInfo ("short * __restrict__", "Fire10_O0",    ARG_SCOPE_LOCAL,  ARG_DIR_INOUT,  AT_MEM_UNDEF, AT_MEM_UNDEF, NULL),
            TCArgInfo ("short * __restrict__", "Fire10_Out",   ARG_SCOPE_LOCAL,  ARG_DIR_INOUT,  AT_MEM_UNDEF, AT_MEM_UNDEF, NULL),

            TCArgInfo ("short * __restrict__", "Fire11_O0",    ARG_SCOPE_LOCAL,  ARG_DIR_INOUT,  AT_MEM_UNDEF, AT_MEM_UNDEF, NULL),
            TCArgInfo ("short * __restrict__", "Fire11_Out",   ARG_SCOPE_LOCAL,  ARG_DIR_INOUT,  AT_MEM_UNDEF, AT_MEM_UNDEF, NULL),

            TCArgInfo ("short * __restrict__", "Fire12_O0",    ARG_SCOPE_LOCAL,  ARG_DIR_INOUT,  AT_MEM_UNDEF, AT_MEM_UNDEF, NULL),
            TCArgInfo ("short * __restrict__", "Fire12_Out",   ARG_SCOPE_LOCAL,  ARG_DIR_INOUT,  AT_MEM_UNDEF, AT_MEM_UNDEF, NULL)
        )
    );

    AddNode("Conv0MP0",
        Bindings(4,
        GNodeArg(GNA_IN,    "NetworkIn", 0),
        GNodeArg(GNA_IN,    "Conv0MP0_W", 0),
        GNodeArg(GNA_IN,    "Conv0MP0_B", 0),
        GNodeArg(GNA_OUT,   "Conv0MP0_Out", 0)));

    AddNode("Conv1MP1",
        Bindings(4,
        GNodeArg(GNA_IN,    "Conv0MP0_Out", 0),
        GNodeArg(GNA_IN,    "Conv1MP1_W", 0),
        GNodeArg(GNA_IN,    "Conv1MP1_B", 0),
        GNodeArg(GNA_OUT,   "Conv1MP1_Out", 0)));

    AddNode("Fire3",
        Bindings(9,
        GNodeArg(GNA_IN,    "Conv1MP1_Out", 0),
        GNodeArg(GNA_IN,    "Fire3_W0", 0),
        GNodeArg(GNA_IN,    "Fire3_B0", 0),
        GNodeArg(GNA_IN,    "Fire3_O0", 0),
        GNodeArg(GNA_IN,    "Fire3_W1", 0),
        GNodeArg(GNA_IN,    "Fire3_B1", 0),
        GNodeArg(GNA_IN,    "Fire3_W2", 0),
        GNodeArg(GNA_IN,    "Fire3_B2", 0),
        GNodeArg(GNA_OUT,   "Fire3_Out", 0)));

    AddNode("Fire4",
        Bindings(9,
        GNodeArg(GNA_IN,    "Fire3_Out", 0),
        GNodeArg(GNA_IN,    "Fire4_W0", 0),
        GNodeArg(GNA_IN,    "Fire4_B0", 0),
        GNodeArg(GNA_IN,    "Fire4_O0", 0),
        GNodeArg(GNA_IN,    "Fire4_W1", 0),
        GNodeArg(GNA_IN,    "Fire4_B1", 0),
        GNodeArg(GNA_IN,    "Fire4_W2", 0),
        GNodeArg(GNA_IN,    "Fire4_B2", 0),
        GNodeArg(GNA_OUT,   "Fire4_Out", 0)));

    AddNode("Fire6",
        Bindings(9,
        GNodeArg(GNA_IN,    "Fire4_Out", 0),
        GNodeArg(GNA_IN,    "Fire6_W0", 0),
        GNodeArg(GNA_IN,    "Fire6_B0", 0),
        GNodeArg(GNA_IN,    "Fire6_O0", 0),
        GNodeArg(GNA_IN,    "Fire6_W1", 0),
        GNodeArg(GNA_IN,    "Fire6_B1", 0),
        GNodeArg(GNA_IN,    "Fire6_W2", 0),
        GNodeArg(GNA_IN,    "Fire6_B2", 0),
        GNodeArg(GNA_OUT,   "Fire6_Out", 0)));

    AddNode("Fire7",
        Bindings(9,
        GNodeArg(GNA_IN,    "Fire6_Out", 0),
        GNodeArg(GNA_IN,    "Fire7_W0", 0),
        GNodeArg(GNA_IN,    "Fire7_B0", 0),
        GNodeArg(GNA_IN,    "Fire7_O0", 0),
        GNodeArg(GNA_IN,    "Fire7_W1", 0),
        GNodeArg(GNA_IN,    "Fire7_B1", 0),
        GNodeArg(GNA_IN,    "Fire7_W2", 0),
        GNodeArg(GNA_IN,    "Fire7_B2", 0),
        GNodeArg(GNA_OUT,   "Fire7_Out", 0)));

    AddNode("Fire9",
        Bindings(9,
        GNodeArg(GNA_IN,    "Fire7_Out", 0),
        GNodeArg(GNA_IN,    "Fire9_W0", 0),
        GNodeArg(GNA_IN,    "Fire9_B0", 0),
        GNodeArg(GNA_IN,    "Fire9_O0", 0),
        GNodeArg(GNA_IN,    "Fire9_W1", 0),
        GNodeArg(GNA_IN,    "Fire9_B1", 0),
        GNodeArg(GNA_IN,    "Fire9_W2", 0),
        GNodeArg(GNA_IN,    "Fire9_B2", 0),
        GNodeArg(GNA_OUT,   "Fire9_Out", 0)));

    AddNode("Fire10",
        Bindings(9,
        GNodeArg(GNA_IN,    "Fire9_Out", 0),
        GNodeArg(GNA_IN,    "Fire10_W0", 0),
        GNodeArg(GNA_IN,    "Fire10_B0", 0),
        GNodeArg(GNA_IN,    "Fire10_O0", 0),
        GNodeArg(GNA_IN,    "Fire10_W1", 0),
        GNodeArg(GNA_IN,    "Fire10_B1", 0),
        GNodeArg(GNA_IN,    "Fire10_W2", 0),
        GNodeArg(GNA_IN,    "Fire10_B2", 0),
        GNodeArg(GNA_OUT,   "Fire10_Out", 0)));

    AddNode("Fire11",
        Bindings(9,
        GNodeArg(GNA_IN,    "Fire10_Out", 0),
        GNodeArg(GNA_IN,    "Fire11_W0", 0),
        GNodeArg(GNA_IN,    "Fire11_B0", 0),
        GNodeArg(GNA_IN,    "Fire11_O0", 0),
        GNodeArg(GNA_IN,    "Fire11_W1", 0),
        GNodeArg(GNA_IN,    "Fire11_B1", 0),
        GNodeArg(GNA_IN,    "Fire11_W2", 0),
        GNodeArg(GNA_IN,    "Fire11_B2", 0),
        GNodeArg(GNA_OUT,   "Fire11_Out", 0)));

    AddNode("Fire12",
        Bindings(9,
        GNodeArg(GNA_IN,    "Fire11_Out", 0),
        GNodeArg(GNA_IN,    "Fire12_W0", 0),
        GNodeArg(GNA_IN,    "Fire12_B0", 0),
        GNodeArg(GNA_IN,    "Fire12_O0", 0),
        GNodeArg(GNA_IN,    "Fire12_W1", 0),
        GNodeArg(GNA_IN,    "Fire12_B1", 0),
        GNodeArg(GNA_IN,    "Fire12_W2", 0),
        GNodeArg(GNA_IN,    "Fire12_B2", 0),
        GNodeArg(GNA_OUT,   "Fire12_Out", 0)));

    AddNode("GPool10",
        Bindings(2,
            GNodeArg(GNA_IN,    "Fire12_Out", 0),
            GNodeArg(GNA_OUT,   "NetworkOut", 0)
        )
    );

    CloseGraph();
#endif
}

int main(int argc, char **argv)
{
    if (TilerParseOptions(argc, argv))
    {
        printf("Failed to initialize or incorrect output directory.\n");
        return -1;
    }

    CnnModel(
        64*1024 - (CL_STACK_SIZE + 7 * CL_SLAVE_STACK_SIZE + 7 * 1024), // 7 KB for local cluster data (5 KB is minimum)
        135*1024,     // Gives best performance for L2 < 150 KB according to tests
        1*1024*1024,  // 1 MB is enough
        16*1024*1024  // Give all HyperFlash as it's not used anywhere else
    );
    GenerateTilingCode();
    return 0;
}
