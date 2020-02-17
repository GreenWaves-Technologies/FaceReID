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

#ifndef __BLE_PROTOCOL_H__
#define __BLE_PROTOCOL_H__

#define BLE_CMD_READ_STRANGER           0x10
#define BLE_CMD_GET_STRANGER_NAME       0x11
#define BLE_CMD_GET_STRANGER_PHOTO      0x12
#define BLE_CMD_GET_STRANGER_DESCRIPTOR 0x13
#define BLE_CMD_DROP_STRANGER           0x14

#define BLE_CMD_READ_VISITOR            0x15
#define BLE_CMD_GET_VISITOR_NAME        0x16
#define BLE_CMD_GET_VISITOR_DESCRIPTOR  0x17
#define BLE_CMD_DROP_VISITOR            0x18

#define BLE_CMD_WRITE          0x20
#define BLE_CMD_SET_NAME       0x21
#define BLE_CMD_SET_DESCRIPTOR 0x22

#define BLE_CMD_ACK            0x33

#define BLE_CMD_EXIT           0x55

#define BLE_CMD_HB             0x56

// Chunk size is maximum DMA transfer size.
// There is no signal from BLE module on actual data transfer and all blocking
// functions just wait for DMA request. If chunk size is larger than maximum DMA
// transfer size several transfers are invoked. Otherwise it can lead to buffers
// overlap and data loss.
#define DATA_CHUNK_SIZE    1024

#endif
