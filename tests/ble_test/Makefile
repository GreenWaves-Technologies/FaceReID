# Copyright 2019 GreenWaves Technologies, SAS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

APP = ble_test

USE_PMSIS_BSP=1
BOARD_NAME = gapoc_a

APP_SRCS += ble_main.c ../../ReID-Demo/BleUserManager.c ../../ReID-Demo/dnn_utils.c ../../ReID-Demo/strangers_db.c ../../ReID-Demo/face_db.c ../../ReID-Demo/ImgIO.c ../../ReID-Demo/display.c $(PLATFORM_SRCS)

APP_CFLAGS += -DUSE_PMSIS=1 -DUSE_BLE_USER_MANAGEMENT=1 -I$(PWD) -I$(PWD)/../../ReID-Demo

PLPBRIDGE_FLAGS += -ftdi -fileIO

include $(RULES_DIR)/pmsis_rules.mk
