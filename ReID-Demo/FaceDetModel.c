/*
 * Copyright (C) 2017-2019 GreenWaves Technologies
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 *
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
