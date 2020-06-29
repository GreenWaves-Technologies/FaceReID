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

REID_PATH := ../../ReID-Demo

APP_SRCS := \
  main.c \
  $(REID_PATH)/face_db.c \
  $(REID_PATH)/dnn_utils.c
APP_INC += $(PWD) $(REID_PATH) "$(TILER_PATH)/include"
APP_CFLAGS += -O3 -DNORM_ROUND

FACE_DB_SIZE := $(shell wc -l < $(REID_PATH)/known_faces/index.txt)
DATA_FILES := $(REID_PATH)/known_faces/index.txt $(wildcard $(REID_PATH)/known_faces/*.bin)

APP_CFLAGS += -DSTATIC_FACE_DB=1 -DFACE_DB_SIZE=$(FACE_DB_SIZE)

ifeq ($(SILENT),1)
  APP_CFLAGS += -DSILENT=1
endif

READFS_FILES += $(realpath $(DATA_FILES))

include $(RULES_DIR)/pmsis_rules.mk
