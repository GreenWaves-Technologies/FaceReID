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
#define MAX_NUM_OUT_WINS 20

#if !defined(__FREERTOS__)
# define PERF_COUNT
#endif

#define HAVE_DISPLAY
#define HAVE_CAMERA

#ifdef SILENT
#define PRINTF(...) ((void) 0)
#else
#define PRINTF printf
#endif  /* DEBUG */

#if defined(CONFIG_GAPOC_A)
#define CAMERA_WIDTH  320
#define CAMERA_HEIGHT 240
#else
#define CAMERA_WIDTH 324
#define CAMERA_HEIGHT 244
#endif

#define CL_SLAVE_STACK_SIZE 1024
#define CL_STACK_SIZE       2048

#define WOUT_INIT 64
#define HOUT_INIT 48

#define LCD_WIDTH  320
#define LCD_HEIGHT 240

#define LCD_ORIENTATION PI_ILI_ORIENTATION_270

#define FACE_DESCRIPTOR_SIZE 512

#define REID_L2_THRESHOLD 175000000 // L2 metric threshold for users identification.

#define STRANGER_L2_THRESHOLD 150000000 // L2 metric threshold for strangers de-duplication.

//Instructions for REID_L2_THRESHOLD and STRANGER_L2_THRESHOLD changing are in documentation/quantization_instruction.md

#define MEMORY_POOL_SIZE 150 * 1024

#define FACE_DETECTOR_STABILIZATION_PERIOD 3

#ifndef BLE_NAME
# define BLE_NAME "GreenWaves-GAPOC"
#endif
#define BLE_TIMEOUT 30000000 // 30 s

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

#endif //SETUP_H
