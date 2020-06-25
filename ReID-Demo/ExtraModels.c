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
#include "setup.h"

// AutoTiler Libraries
#include "AutoTilerLib.h"

void GenerateResizeShort(char *Name, int Wi, int Hi, int Wo, int Ho)
{
    UserKernel(Name,
        KernelIterSpace(1, IterTiledSpace(KER_ITER_TILE0)),
        TILE_HOR,
        CArgs(2, TCArg("unsigned char *", "In"), TCArg("short *", "Out")),
        Calls(1, Call("KerResizeBilinearShort", LOC_LOOP,
            Bindings(8, K_Arg("In", KER_ARG_TILE),
                        K_Arg("In", KER_ARG_W),
                        K_Arg("In", KER_ARG_H),
                        K_Arg("Out", KER_ARG_TILE),
                        K_Arg("Out", KER_ARG_W),
                        K_Arg("Out", KER_ARG_H),
                        K_Arg("Out", KER_ARG_TILE_H),
                        K_Arg("In", KER_ARG_TILE_BASE)))),
        KerArgs(2,
            KerArg("In",  KerArgSpace(1,KER_ITER_TILE0), OBJ_IN_DB,  Wi, Hi, sizeof(char), 1, OBJ_CONSTRAINTS_DYNAMIC, 0, "In"),
            KerArg("Out", KerArgSpace(1,KER_ITER_TILE0), OBJ_OUT_DB, Wo, Ho, sizeof(short), 0, OBJ_CONSTRAINTS_DYNAMIC, 0, "Out")
        )
    );
}


int main(int argc, char **argv)
{
    // This will parse AutoTiler options and perform various initializations
    if (TilerParseOptions(argc, argv))
    {
        printf("Failed to initialize or incorrect output arguments directory.\n");
        return 1;
    }

    SetInlineMode(ALWAYS_INLINE);
    SetSymbolNames("ExtraKernels_L1_Memory", "ExtraKernels_L2_Memory");
    SetSymbolDynamics();
    SetKernelOpts(KER_OPT_NONE, KER_OPT_BUFFER_PROMOTE);

    SetUsedFilesNames(0, 1, "ExtraBasicKernels.h");
    SetGeneratedFilesNames("ExtraKernels.c", "ExtraKernels.h");

    SetL1MemorySize(64*1024 - (CL_STACK_SIZE + 7 * CL_SLAVE_STACK_SIZE + 7 * 1024));

    LibKernel("KerResizeBilinearShort", CALL_PARALLEL,
        CArgs(8,
            TCArg("unsigned char * __restrict__", "In"),
            TCArg("unsigned int", "Win"),
            TCArg("unsigned int", "Hin"),
            TCArg("short * __restrict__", "Out"),
            TCArg("unsigned int", "Wout"),
            TCArg("unsigned int", "Hout"),
            TCArg("unsigned int", "HTileOut"),
            TCArg("unsigned int", "FirstLineIndex")),
        "KerResizeBilinearShort_ArgT", NULL
    );

    GenerateResizeShort("ResizeImageForDnn_Scale1", 152, 152, 128, 128);
    GenerateResizeShort("ResizeImageForDnn_Scale2", 194, 194, 128, 128);

    // Now that we are done with model parsing we generate the code
    GenerateTilingCode();
    return 0;
}
