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

#ifndef __BLE_PROTOCAL_H__
#define __BLE_PROTOCAL_H__

#define BLE_READ           0x10
#define BLE_GET_NAME       0x11
#define BLE_GET_PHOTO      0x12
#define BLE_GET_DESCRIPTOR 0x13
#define BLE_REMOVE         0x14

#define BLE_WRITE          0x20
#define BLE_SET_NAME       0x21
#define BLE_SET_DESCRIPTOR 0x22

#define BLE_ACK            0x33

#define BLE_EXIT           0x55

// Chank size is maximum DMA transfer size
// There is no sygnal from BLE module on actual data transfer and all blocking
// functions just wait for DMA reqest if chunk size is larger than maximum DMA
// trasnfer size several transfers are invoked that can lead to buffers overlap
// and data loss
#define DATA_CHANK_SIZE    1024

#endif
