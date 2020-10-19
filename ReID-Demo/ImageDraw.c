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

#include "ImageDraw.h"

#define Max(a, b) (((a)>(b))?(a):(b))
#define Min(a, b) (((a)<(b))?(a):(b))

void DrawRectangle(unsigned char *Img, int W, int H, int x, int y, int w, int h, unsigned char Value)
{
    int x0, x1, y0, y1;

    y0 = Max(Min(y, H - 1), 0);
    y1 = Max(Min(y + h - 1, H - 1), 0);

    x0 = x;
    if (x0 >= 0 && x0 < W) {
        for (int i = y0; i <= y1; i++)
            Img[i * W + x0] = Value;
    }

    x1 = x + w - 1;
    if (x1 >= 0 && x1 < W) {
        for (int i = y0; i <= y1; i++)
            Img[i * W + x1] = Value;
    }

    x0 = Max(Min(x, W - 1), 0);
    x1 = Max(Min(x + w - 1, W - 1), 0);

    y0 = y;
    if (y0 >= 0 && y0 < H) {
        for (int i = x0; i <= x1; i++)
            Img[y0 * W + i] = Value;
    }

    y1 = y + h - 1;
    if (y1 >= 0 && y1 < H) {
        for (int i = x0; i <= x1; i++)
            Img[y1 * W + i] = Value;
    }
}
