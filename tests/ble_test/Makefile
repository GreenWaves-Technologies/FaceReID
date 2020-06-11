# Copyright 2019-2020 GreenWaves Technologies, SAS
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

USE_PMSIS_BSP=1
export GAP_USE_OPENOCD=1
io=host

REID_PATH := ../../ReID-Demo

APP := ble_test
APP_SRCS := \
  ble_main.c \
  $(REID_PATH)/BleUserManager.c \
  $(REID_PATH)/dnn_utils.c \
  $(REID_PATH)/strangers_db.c \
  $(REID_PATH)/face_db.c \
  $(REID_PATH)/ImgIO.c \
  $(REID_PATH)/display.c
APP_INC += $(PWD) $(REID_PATH)
APP_CFLAGS += -DUSE_BLE_USER_MANAGEMENT=1
ifeq ($(SILENT),1)
  APP_CFLAGS += -DSILENT=1
endif

include $(RULES_DIR)/pmsis_rules.mk
