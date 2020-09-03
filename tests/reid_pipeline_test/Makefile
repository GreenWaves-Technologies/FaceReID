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

GRAPH := 1
STATIC_FACE_DB := 1
USE_PMSIS_BSP := 1

REID_PATH := ../../ReID-Demo
CNN_LAYERS_PATH := $(REID_PATH)/quantized_model
CNN_LAYERS_SRC := $(wildcard $(CNN_LAYERS_PATH)/*.json)
CNN_LAYERS_BIN := $(CNN_LAYERS_SRC:.json=.weights.bin) $(CNN_LAYERS_SRC:.json=.bias.bin)

%.bias.bin %.weights.bin: %.json
	../../scripts/convert2gap.py $< $(basename $<)

REID_MODEL_GEN := CnnKernels
REID_MODEL_GEN_C := $(addsuffix .c, $(REID_MODEL_GEN))
REID_MODEL_GEN_CH := $(REID_MODEL_GEN_C) $(addsuffix .h, $(REID_MODEL_GEN))

REID_GEN_SRCS := \
  $(REID_PATH)/ReIDModel.c \
  $(REID_PATH)/FireGenerator.c \
  $(REID_PATH)/layer_params.c \
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

GenExtraTile: $(REID_PATH)/ExtraModels.c
	gcc -o $@ -I"$(TILER_INC)" -I"$(TILER_CNN_GENERATOR_PATH)" $(REID_PATH)/ExtraModels.c $(TILER_LIB)

$(EXTRA_MODEL_GEN_C): GenExtraTile
	./GenExtraTile

extra_model: $(EXTRA_MODEL_GEN_C)

tiler_models: reid_model extra_model

APP_SRCS := \
  main.c \
  $(REID_PATH)/reid_pipeline.c \
  $(REID_PATH)/dnn_utils.c \
  $(REID_PATH)/face_db.c \
  $(REID_PATH)/ExtraBasicKernels.c \
  $(REID_PATH)/ImgIO.c \
  $(REID_MODEL_GEN_C) \
  $(EXTRA_MODEL_GEN_C) \
  $(TILER_CNN_KERNEL_PATH)/CNN_Conv_BasicKernels.c \
  $(TILER_CNN_KERNEL_PATH)/CNN_BiasReLULinear_BasicKernels.c \
  $(TILER_CNN_KERNEL_PATH)/CNN_Pooling_BasicKernels.c

ifeq ($(GRAPH),1)
  APP_SRCS += $(REID_PATH)/network_process_graph.c
else
  APP_SRCS += \
    $(REID_PATH)/network_process_manual.c \
    $(REID_PATH)/layer_params.c
endif

APP_INC += $(PWD) $(REID_PATH) "$(TILER_INC)" "$(TILER_CNN_KERNEL_PATH)"
APP_CFLAGS += -O3 -DNORM_ROUND -DCONFIG_GAPOC_A
ifeq ($(GRAPH),1)
  APP_CFLAGS += -DGRAPH
endif
ifeq ($(SILENT),1)
  APP_CFLAGS += -DSILENT=1
endif

FACE_DB_SIZE := $(shell wc -l < $(REID_PATH)/known_faces/index.txt)
READFS_FILES += $(REID_PATH)/known_faces/index.txt $(wildcard $(REID_PATH)/known_faces/*.bin)
APP_CFLAGS += -DSTATIC_FACE_DB=1 -DFACE_DB_SIZE=$(FACE_DB_SIZE)

ifeq ($(GRAPH),1)
  READFS_FILES += $(CNN_DATA_BIN)
else
  READFS_FILES += $(CNN_LAYERS_BIN)
endif

build: tiler_models

clean::
	rm -f GenReidNet GenExtraTile
	rm -f $(REID_MODEL_GEN_CH) $(EXTRA_MODEL_GEN_CH)

.PHONY: reid_model extra_model tiler_models build clean

include $(RULES_DIR)/pmsis_rules.mk
