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

#include <stdio.h>
#include <math.h>
#include "pmsis.h"
#include "ExtraBasicKernels.h"

#define Min(a, b)               (((a)<(b))?(a):(b))

static inline unsigned int __attribute__((always_inline)) ChunkSize(unsigned int X)
{
    unsigned int NCore;
    unsigned int Log2Core;
    unsigned int Chunk;

    NCore = pi_cl_cluster_nb_cores();
    Log2Core = gap_fl1(NCore);
    Chunk = (X>>Log2Core) + ((X&(NCore-1))!=0);
    return Chunk;
}

void KerResizeBilinearShort(KerResizeBilinearShort_ArgT *Arg)
{
    unsigned char * __restrict__ In  = Arg->In;
    unsigned int Win                 = Arg->Win;
    unsigned int Hin                 = Arg->Hin;
    short * __restrict__ Out         = Arg->Out;
    unsigned int Wout                = Arg->Wout;
    unsigned int Hout                = Arg->Hout;
    unsigned int HTileOut            = Arg->HTileOut;
    unsigned int FirstLineIndex      = Arg->FirstLineIndex;

    unsigned int CoreId = gap_coreid();
    unsigned int ChunkCell = ChunkSize(Wout);
    unsigned int First = CoreId*ChunkCell, Last  = Min(Wout, First+ChunkCell);

    unsigned int WStep = ((Win-1)<<16)/Wout;
    unsigned int HStep = ((Hin-1)<<16)/Hout;

    unsigned int x, y;
    unsigned int hCoeff = HStep*FirstLineIndex;
    unsigned int BaseY = (hCoeff>>16);
    for (y = 0 ; y < HTileOut ; y++) {
            unsigned int offsetY = (hCoeff >> 16) - BaseY;
            unsigned int hc2 = (hCoeff >> 9) & 127;
            unsigned int hc1 = 128 - hc2;
            // unsigned int wCoeff = 0;
            unsigned int wCoeff = First*WStep;

            // for (x = 0 ; x < Wout ; x++) {
            for (x = First ; x < Last ; x++) {
                    unsigned int offsetX = (wCoeff >> 16);
                    unsigned int wc2 = (wCoeff >> 9) & 127;
                    unsigned int wc1 = 128 - wc2;
                    unsigned int P1 = In[offsetY*Win       + offsetX    ];
                    unsigned int P2 = In[(offsetY + 1)*Win + offsetX    ];
                    unsigned int P3 = In[offsetY*Win       + offsetX + 1];
                    unsigned int P4 = In[(offsetY + 1)*Win + offsetX + 1];

                    unsigned char tmp = ((P1*hc1 + P2*hc2)*wc1 + (P3*hc1 + P4*hc2)*wc2) >> 14;
                    Out[y*Wout + x] = tmp;
                    wCoeff += WStep;
            }
            hCoeff += HStep;
    }
    gap_waitbarrier(0);
}
