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

USE_PMSIS_BSP=1
DEVICE_CONNECTION ?= -jtag

APP_SRCS = main.c ../../ReID-Demo/face_db.c ../../ReID-Demo/dnn_utils.c

APP_CFLAGS += -I$(PWD) -I$(PWD)/../../ReID-Demo  -I"$(TILER_PATH)/include" -I"$(CNN_KER_PATH)/include"

FACE_DB_SIZE=$(shell wc -l < ./known_faces/index.txt)
DATA_FILES = ./known_faces/index.txt ./known_faces/*.bin

APP_CFLAGS += -DSTATIC_FACE_DB=1 -DFACE_DB_SIZE=$(FACE_DB_SIZE)

PLPBRIDGE_FLAGS += -f $(DATA_FILES) $(DEVICE_CONNECTION) -fileIO 10

.PHONY: all clean

include $(GAP_SDK_HOME)/tools/rules/pmsis_rules.mk
