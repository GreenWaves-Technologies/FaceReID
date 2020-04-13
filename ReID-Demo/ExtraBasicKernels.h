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

#ifndef EXTRABASICKERNELS_H__
#define EXTRABASICKERNELS_H__

#include "Gap.h"

typedef struct {
	unsigned char * __restrict__ In;
	unsigned int Win;
	unsigned int Hin;
	short * __restrict__ Out;
	unsigned int Wout;
	unsigned int Hout;
	unsigned int HTileOut;
	unsigned int FirstLineIndex;
} KerResizeBilinearShort_ArgT;

void KerResizeBilinearShort(KerResizeBilinearShort_ArgT *Arg);

#endif // EXTRABASICKERNELS_H__
