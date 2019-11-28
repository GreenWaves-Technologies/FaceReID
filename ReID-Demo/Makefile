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

#BOARD_NAME=gapoc_a
#SILENT=0
#STATIC_FACE_DB=0
#BLE_NOTIFIER=0

USE_PMSIS_BSP=1
ifeq ($(BOARD_NAME),gapoc_a)
  DEVICE_CONNECTION = -ftdi
else
  DEVICE_CONNECTION = -jtag
endif

TILER_PATH = $(GAP_SDK_HOME)/tools/autotiler_v3
TILER_LIB = $(TILER_PATH)/lib/libtile.a
TILER_GENERATOR_PATH = $(TILER_PATH)/generators
CNN_GEN_PATH = $(TILER_GENERATOR_PATH)/CNN
CNN_KER_PATH = $(TILER_GENERATOR_PATH)/CNN
CNN_KERNELS_SRC = $(wildcard $(CNN_KER_PATH)/CNN_*.c)
CNN_LAYERS_SRC=$(wildcard quantized_model/*.json)
ifeq ($(ALREADY_FLASHED),1)
	CNN_LAYERS_BIN=
else
	CNN_LAYERS_BIN=$(CNN_LAYERS_SRC:.json=.bin)
endif

DETECTION_MODEL_SRC = FaceDetGenerator.c FaceDetModel.c
DETECTION_MODEL_GEN = FaceDetKernels
DETECTION_MODEL_GEN_C = $(addsuffix .c, $(DETECTION_MODEL_GEN))
DETECTION_MODEL_GEN_CLEAN = $(DETECTION_MODEL_GEN_C) $(addsuffix .h, $(DETECTION_MODEL_GEN))

ifeq ($(BOARD_NAME),gapoc_a)
  TILER_PLATFORM_FLAGS=-D_FOR_GAPOC_=1
endif

GenDetectionTile: $(DETECTION_MODEL_SRC)
	gcc -no-pie -o GenDetectionTile -D_FOR_GAPOC_=1 -I"$(TILER_INC)" $(DETECTION_MODEL_SRC) $(TILER_LIB)

$(DETECTION_MODEL_GEN_C): GenDetectionTile
	./GenDetectionTile

detection_model: $(DETECTION_MODEL_GEN_C)

%.bin: %.json
	../scripts/convert2gap.py $< $(basename $@)

REID_MODEL_GEN = CnnKernels
REID_MODEL_GEN_C = $(addsuffix .c, $(REID_MODEL_GEN))
REID_MODEL_GEN_CLEAN = $(REID_MODEL_GEN_C) $(addsuffix .h, $(REID_MODEL_GEN))

GenReidNet: ReIDModel.c param_layer_manual.c norm_list.h
	gcc -no-pie -o GenReidNet -I"$(TILER_INC)" -I"$(CNN_GEN_PATH)" ReIDModel.c param_layer_manual.c $(CNN_GEN_PATH)/CNN_Generators.c $(TILER_LIB)

$(REID_MODEL_GEN_C): GenReidNet
	./GenReidNet

reid_model: $(REID_MODEL_GEN_C) $(CNN_LAYERS_BIN)

EXTRA_MODEL_GEN = ExtraKernels
EXTRA_MODEL_GEN_C = $(addsuffix .c, $(EXTRA_MODEL_GEN))
EXTRA_MODEL_GEN_CLEAN = $(EXTRA_MODEL_GEN_C) $(addsuffix .h, $(EXTRA_MODEL_GEN))

GenExtraTile: ExtraModels.c
	gcc -no-pie -o GenExtraTile -I"$(TILER_INC)" -I"$(CNN_GEN_PATH)" ExtraModels.c $(TILER_LIB)

$(EXTRA_MODEL_GEN_C): GenExtraTile
	./GenExtraTile

extra_model: $(EXTRA_MODEL_GEN_C)

tiler_models: detection_model reid_model extra_model $(CNN_LAYERS_BIN)

APP_SRCS = main.c cascade.c facedet_pipeline.c reid_pipeline.c FaceDetBasicKernels.c ImageDraw.c ImgIO.c network_process_manual.c param_layer_manual.c dnn_utils.c face_db.c ExtraBasicKernels.c $(DETECTION_MODEL_GEN_C) $(REID_MODEL_GEN_C) $(CNN_KERNELS_SRC) $(EXTRA_MODEL_GEN_C)

ifeq ($(BOARD_NAME),gapoc_a)
  APP_CFLAGS += -D_FOR_GAPOC_=1
endif

DATA_FILES = ./quantized_model/*.bin
APP_CFLAGS += -Os -I$(PWD) -I"$(TILER_INC)" -I"$(CNN_KER_PATH)"

ifeq ($(STATIC_FACE_DB),1)
  ifeq ($(BLE_NOTIFIER),1)
    APP_SRCS += StaticUserManagerBleNotifier.c
    APP_CFLAGS += -DBLE_NOTIFIER=1
  else
    APP_SRCS += StaticUserManager.c
  endif
  FACE_DB_SIZE=$(shell wc -l < ./known_faces/index.txt)
  DATA_FILES += ./known_faces/index.txt ./known_faces/*.bin
  APP_CFLAGS += -DSTATIC_FACE_DB=1 -DFACE_DB_SIZE=$(FACE_DB_SIZE)
else
  ifeq ($(BOARD_NAME),gapoc_a)
    APP_SRCS += BleUserManager.c strangers_db.c
    APP_CFLAGS += -DUSE_BLE_USER_MANAGEMENT=1
  else
    APP_SRCS += KeyboardUserManager.c ui_input_box.c PS2Keyboard.c
    APP_CFLAGS += -DUSE_PS2_KEYBOARD=1
  endif
endif

ifeq ($(DUMP_SUCCESSFUL_FRAME),1)
  APP_CFLAGS += -DDUMP_SUCCESSFUL_FRAME=1
endif

ifeq ($(ALREADY_FLASHED),1)
  PLPBRIDGE_FLAGS += $(DEVICE_CONNECTION) #-fileIO 10
else
  PLPBRIDGE_FLAGS += -f $(DATA_FILES) $(DEVICE_CONNECTION) #-fileIO 10
endif

ifeq ($(SILENT),1)
  APP_CFLAGS += -DSILENT=1
  PLPBRIDGE_FLAGS += -hyper
else
  PLPBRIDGE_FLAGS += -fileIO 10
endif

ifeq ($(DUMP_SUCCESSFUL_FRAME),1)
prepare_debug:
	mkdir -p dumps
	rm -rf dumps/*
else
prepare_debug:
	true
endif

all:: tiler_models

run:: prepare_debug

clean::
	rm -rf BUILD
	rm -rf GenDetectionTile
	rm -rf GenReidNet
	rm -rf GenExtraTile
	rm -rf $(REID_MODEL_GEN_CLEAN) $(EXTRA_MODEL_GEN_CLEAN) $(DETECTION_MODEL_GEN_CLEAN)

.PHONY: detection_model reid_model extra_model tiler_models all clean

include $(GAP_SDK_HOME)/tools/rules/pmsis_rules.mk
