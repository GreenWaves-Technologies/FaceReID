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

TILER_PATH = $(GAP_SDK_HOME)/tools/autotiler_v3
TILER_LIB = $(TILER_PATH)/lib/libtile.a
TILER_GENERATOR_PATH = $(TILER_PATH)/generators
CNN_GEN_PATH = $(TILER_GENERATOR_PATH)/CNN
CNN_KER_PATH = $(TILER_GENERATOR_PATH)/CNN
CNN_KERNELS_SRC = $(wildcard $(CNN_KER_PATH)/CNN_*.c)

REID_MODEL_GEN = CnnKernels
REID_MODEL_GEN_C = $(addsuffix .c, $(REID_MODEL_GEN))
REID_MODEL_GEN_CLEAN = $(REID_MODEL_GEN_C) $(addsuffix .h, $(REID_MODEL_GEN))

REID_GEN_SRCS = \
  ../../ReID-Demo/ReIDModel.c \
  ../../ReID-Demo/param_layer_manual.c \
  $(CNN_GEN_PATH)/CNN_Generators.c

GenReidNet: $(REID_GEN_SRCS)
	gcc -no-pie -o GenReidNet -DGRAPH -DNO_CONST -DCONFIG_GAPOC_A -I"$(TILER_PATH)/include" -I"$(CNN_GEN_PATH)" $(REID_GEN_SRCS) $(TILER_LIB)

$(REID_MODEL_GEN_C): GenReidNet
	./GenReidNet

reid_model: $(REID_MODEL_GEN_C) $(CNN_LAYERS_BIN)

EXTRA_MODEL_GEN = ExtraKernels
EXTRA_MODEL_GEN_C = $(addsuffix .c, $(EXTRA_MODEL_GEN))
EXTRA_MODEL_GEN_CLEAN = $(EXTRA_MODEL_GEN_C) $(addsuffix .h, $(EXTRA_MODEL_GEN))

GenExtraTile: ../../ReID-Demo/ExtraModels.c
	gcc -no-pie -o GenExtraTile -I"$(TILER_INC)" -I"$(CNN_GEN_PATH)" ../../ReID-Demo/ExtraModels.c $(TILER_LIB)

$(EXTRA_MODEL_GEN_C): GenExtraTile
	./GenExtraTile

extra_model: $(EXTRA_MODEL_GEN_C)

APP_SRCS = \
  main.c \
  ../../ReID-Demo/network_process_manual.c \
  ../../ReID-Demo/param_layer_manual.c \
  ../../ReID-Demo/dnn_utils.c \
  ../../ReID-Demo/ImgIO.c \
  $(REID_MODEL_GEN_C) \
  $(CNN_KERNELS_SRC)
APP_INC += $(PWD) $(PWD)/../../ReID-Demo "$(TILER_INC)" "$(CNN_KER_PATH)"
APP_CFLAGS += $(CONTROL_MACRO) -DNORM_ROUND=1 -Os
ifeq ($(SILENT),1)
  APP_CFLAGS += -DSILENT=1
endif

PLPBRIDGE_FLAGS += -f
DATA_FILES = $(wildcard ../../ReID-Demo/quantized_model/*.bin)
READFS_FILES += $(realpath $(DATA_FILES))

tiler_models: reid_model extra_model

build: tiler_models

clean::
	rm -f GenReidNet GenExtraTile
	rm -f output.bin
	rm -f $(REID_MODEL_GEN_CLEAN) $(EXTRA_MODEL_GEN_CLEAN)

.PHONY: reid_model extra_model tiler_models build clean

include $(RULES_DIR)/pmsis_rules.mk
