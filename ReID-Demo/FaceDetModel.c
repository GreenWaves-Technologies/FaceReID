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

// AutoTiler Libraries
#include "AutoTilerLib.h"
// HOG generator
#include "FaceDetGenerator.h"

#include "setup.h"

int main(int argc, char **argv)
{
    // This will parse AutoTiler options and perform various initializations
    if (TilerParseOptions(argc, argv))
    {
        printf("Failed to initialize or incorrect output arguments directory.\n");
        return 1;
    }
    // Setup AutTiler configuration. Used basic kernel libraries, C names to be used for code generation,
    // compilation options, and amount of shared L1 memory that the AutoTiler can use, here 51200 bytes
    FaceDetectionConfiguration(25000);
    // Load the HOG basic kernels template library
    LoadFaceDetectionLibrary();
    // Call HOG generator, here image is [644x482], the HOG parameters come from HOGParameters.h
    unsigned int W = CAMERA_WIDTH, H = CAMERA_HEIGHT;
    unsigned int Wout = WOUT_INIT, Hout = HOUT_INIT;

    printf("Level1: %d x %d\n",Wout,Hout);
    GenerateResize("ResizeImage_1", W, H, Wout, Hout);
    GenerateIntegralImage("IntegralImage_1", Wout, Hout);
    GenerateSquaredIntegralImage("SquaredIntegralImage_1", Wout, Hout);
    GenerateCascadeClassifier("Cascade_1",Wout,Hout,24,24);


    Wout /= 1.25, Hout /= 1.25;
    printf("Level2: %d x %d\n",Wout,Hout);
    GenerateResize("ResizeImage_2", W, H, Wout, Hout);
    GenerateIntegralImage("IntegralImage_2", Wout, Hout);
    GenerateSquaredIntegralImage("SquaredIntegralImage_2", Wout, Hout);
    GenerateCascadeClassifier("Cascade_2",Wout,Hout,24,24);

    Wout /= 1.25, Hout /= 1.25;
    printf("Level3: %d x %d\n",Wout,Hout);
    GenerateResize("ResizeImage_3", W, H, Wout, Hout);
    GenerateIntegralImage("IntegralImage_3", Wout, Hout);
    GenerateSquaredIntegralImage("SquaredIntegralImage_3", Wout, Hout);
    GenerateCascadeClassifier("Cascade_3",Wout,Hout,24,24);

    // Now that we are done with model parsing we generate the code
    GenerateTilingCode();
    return 0;
}
