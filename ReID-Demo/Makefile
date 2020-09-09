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

GRAPH := 1
USE_PMSIS_BSP := 1

CNN_LAYERS_PATH := quantized_model
CNN_LAYERS_SRC := $(wildcard $(CNN_LAYERS_PATH)/*.json)
CNN_LAYERS_BIN := $(CNN_LAYERS_SRC:.json=.weights.bin) $(CNN_LAYERS_SRC:.json=.bias.bin)

DETECTION_MODEL_GEN := FaceDetKernels
DETECTION_MODEL_GEN_C := $(addsuffix .c, $(DETECTION_MODEL_GEN))
DETECTION_MODEL_GEN_CH := $(DETECTION_MODEL_GEN_C) $(addsuffix .h, $(DETECTION_MODEL_GEN))

DETECTION_GEN_SRC := FaceDetGenerator.c FaceDetModel.c
DETECTION_GEN_CFLAGS += -I"$(TILER_INC)"
ifeq ($(BOARD_NAME),gapoc_a)
  DETECTION_GEN_CFLAGS += -DCONFIG_GAPOC_A
endif

GenDetectionTile: $(DETECTION_GEN_SRC)
	gcc -o $@ $(DETECTION_GEN_CFLAGS) $(DETECTION_GEN_SRC) $(TILER_LIB)

$(DETECTION_MODEL_GEN_C): GenDetectionTile
	./GenDetectionTile

detection_model: $(DETECTION_MODEL_GEN_C)

%.bias.bin %.weights.bin: %.json
	../scripts/convert2gap.py $< $(basename $<)

REID_MODEL_GEN := CnnKernels
REID_MODEL_GEN_C := $(addsuffix .c, $(REID_MODEL_GEN))
REID_MODEL_GEN_CH := $(REID_MODEL_GEN_C) $(addsuffix .h, $(REID_MODEL_GEN))

REID_GEN_SRCS := \
  ReIDModel.c \
  FireGenerator.c \
  layer_params.c \
  $(TILER_CNN_GENERATOR_PATH)/CNN_Generators.c \
  $(TILER_CNN_GENERATOR_PATH)/CNN_Generator_Util.c
REID_GEN_CFLAGS += -fcommon -I"$(TILER_INC)" -I"$(TILER_CNN_GENERATOR_PATH)"
ifeq ($(GRAPH),1)
  REID_GEN_CFLAGS += -DGRAPH -DCNN_LAYERS_PATH='"$(CNN_LAYERS_PATH)"'
  CNN_DATA_BIN := SqueezeNet_L3_Flash_Const.dat
endif

GenReidNet: $(REID_GEN_SRCS)
	gcc -o $@ $(REID_GEN_CFLAGS) $(REID_GEN_SRCS) $(TILER_LIB)

$(subst .,%,$(REID_MODEL_GEN_CH) $(CNN_DATA_BIN)): GenReidNet $(CNN_LAYERS_BIN)
	./GenReidNet
	sed -i '/AT_L[12]_ALLOC/d' $(REID_MODEL_GEN_C)
	sed -i '/AT_L[12]_FREE/d'  $(REID_MODEL_GEN_C)

reid_model: $(REID_MODEL_GEN_C) $(CNN_LAYERS_BIN) $(CNN_DATA_BIN)

EXTRA_MODEL_GEN := ExtraKernels
EXTRA_MODEL_GEN_C := $(addsuffix .c, $(EXTRA_MODEL_GEN))
EXTRA_MODEL_GEN_CH := $(EXTRA_MODEL_GEN_C) $(addsuffix .h, $(EXTRA_MODEL_GEN))

GenExtraTile: ExtraModels.c
	gcc -o $@ -I"$(TILER_INC)" -I"$(TILER_CNN_GENERATOR_PATH)" ExtraModels.c $(TILER_LIB)

$(EXTRA_MODEL_GEN_C): GenExtraTile
	./GenExtraTile

extra_model: $(EXTRA_MODEL_GEN_C)

tiler_models: detection_model reid_model extra_model

APP_SRCS := \
  main.c \
  cascade.c \
  display.c \
  face_db.c \
  facedet_pipeline.c \
  reid_pipeline.c \
  FaceDetBasicKernels.c \
  ImageDraw.c \
  ImgIO.c \
  dnn_utils.c \
  ExtraBasicKernels.c \
  $(DETECTION_MODEL_GEN_C) \
  $(REID_MODEL_GEN_C) \
  $(EXTRA_MODEL_GEN_C) \
  $(TILER_CNN_KERNEL_PATH)/CNN_Conv_BasicKernels.c \
  $(TILER_CNN_KERNEL_PATH)/CNN_BiasReLULinear_BasicKernels.c \
  $(TILER_CNN_KERNEL_PATH)/CNN_Pooling_BasicKernels.c

ifeq ($(GRAPH),1)
  APP_SRCS += network_process_graph.c
else
  APP_SRCS += \
    network_process_manual.c \
    layer_params.c
endif

APP_INC += "$(PWD)" "$(TILER_INC)" "$(TILER_CNN_KERNEL_PATH)"

APP_CFLAGS += -Os -flto -DNORM_ROUND
APP_LDFLAGS += -flto

ifeq ($(GRAPH),1)
  APP_CFLAGS += -DGRAPH
endif

ifneq ($(BOARD_NAME),gapoc_a)
  STATIC_FACE_DB := 1
endif

ifeq ($(STATIC_FACE_DB),1)
  FACE_DB_SIZE := $(shell wc -l < known_faces/index.txt)
  ifeq ($(FACE_DB_SIZE),)
    $(error known_faces/index.txt file doesn't exist! Please add people names there)
  endif
  ifeq ($(BLE_NOTIFIER),1)
    APP_SRCS += StaticUserManagerBleNotifier.c
    APP_CFLAGS += -DBLE_NOTIFIER=1
  else
    APP_SRCS += StaticUserManager.c
  endif

  READFS_FILES += known_faces/index.txt $(wildcard known_faces/*.bin)
  APP_CFLAGS += -DSTATIC_FACE_DB=1 -DFACE_DB_SIZE=$(FACE_DB_SIZE)
else
  ifeq ($(BOARD_NAME),gapoc_a)
    APP_SRCS += BleUserManager.c strangers_db.c
    APP_CFLAGS += -DUSE_BLE_USER_MANAGEMENT=1
  endif
  ifneq ($(BLE_NAME),)
    APP_CFLAGS += -DBLE_NAME='"$(BLE_NAME)"'
  endif
endif

ifeq ($(GRAPH),1)
  READFS_FILES += $(CNN_DATA_BIN)
else
  READFS_FILES += $(CNN_LAYERS_BIN)
endif

ifeq ($(SILENT),1)
  APP_CFLAGS += -DSILENT=1
endif

ifeq ($(DUMP_SUCCESSFUL_FRAME),1)
  APP_CFLAGS += -DDUMP_SUCCESSFUL_FRAME=1

prepare_debug:
	mkdir -p dumps
	rm -rf dumps/*
else
prepare_debug:
	true
endif

build: tiler_models

run: prepare_debug

clean::
	rm -f GenReidNet GenExtraTile GenDetectionTile
	rm -f $(REID_MODEL_GEN_CH) $(EXTRA_MODEL_GEN_CH) $(DETECTION_MODEL_GEN_CH)

.PHONY: detection_model reid_model extra_model tiler_models build clean

include $(RULES_DIR)/pmsis_rules.mk
