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

#ifndef SETUP_H
#define SETUP_H

//Cascade stride
#define MAX_NUM_OUT_WINS 15

#if !defined(__FREERTOS__)
# define PERF_COUNT
#endif

#define HAVE_DISPLAY 1
#define HAVE_CAMERA

#ifdef SILENT
#define PRINTF(...) ((void) 0)
#else
#define PRINTF printf
#endif  /* DEBUG */

#if defined(_FOR_GAPOC_)
#define CAMERA_WIDTH    (((640/2)/4)*4)
#define CAMERA_HEIGHT   (((480/2)/4)*4)
#else
#define CAMERA_WIDTH 324
#define CAMERA_HEIGHT 244
#endif

#define CLUSTER_STACK_SIZE 2*1024

#define WOUT_INIT 64
#define HOUT_INIT 48

#define LCD_OFF_X 40
#define LCD_OFF_Y 60

#define FACE_DESCRIPTOR_SIZE 512

#define REID_L2_THRESHOLD 1000000

#define STRANGER_L2_THRESHOLD 1000000

#define MEMORY_POOL_SIZE 140000

#define FACE_DETECTOR_STABILIZATION_PERIOD 3

#if defined(USE_BLE_USER_MANAGEMENT)
# define STRANGERS_DB_SIZE 10
# ifndef STATIC_FACE_DB
#  define FACE_DB_SIZE 10
# endif
#else
# ifndef STATIC_FACE_DB
#  define FACE_DB_SIZE 20
# endif
#endif

#define BUTTON_FUNCTION_PIN 25 // button pin id for pi_pad_init call on Gapoc A
#define BUTTON_PIN_ID 19 // button pin id for pi_gpio_ calls on Gapoc A

// Temporary headers for not exported display functions
void writeFillRect(struct pi_device *device, unsigned short x, unsigned short y, unsigned short w, unsigned short h, unsigned short color);
void setCursor(struct pi_device *device, signed short x, signed short y);
void writeText(struct pi_device *device, char* str,int fontsize);

#endif //SETUP_H
