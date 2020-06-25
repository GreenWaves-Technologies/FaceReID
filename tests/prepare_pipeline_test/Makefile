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

TEST_LEVEL ?= 0

REID_PATH := ../../ReID-Demo
TILER_PATH := $(GAP_SDK_HOME)/tools/autotiler_v3
TILER_LIB := $(TILER_PATH)/lib/libtile.a
TILER_GENERATOR_PATH := $(TILER_PATH)/generators
CNN_GEN_PATH := $(TILER_GENERATOR_PATH)/CNN
CNN_KER_PATH := $(TILER_GENERATOR_PATH)/CNN

REID_MODEL_GEN := CnnKernels
REID_MODEL_GEN_C := $(addsuffix .c, $(REID_MODEL_GEN))
REID_MODEL_GEN_CH := $(REID_MODEL_GEN_C) $(addsuffix .h, $(REID_MODEL_GEN))

REID_GEN_SRCS := \
  $(REID_PATH)/ReIDModel.c \
  $(REID_PATH)/FireGenerator.c \
  $(REID_PATH)/param_layer_manual.c \
  $(CNN_GEN_PATH)/CNN_Generators.c \
  $(CNN_GEN_PATH)/CNN_Generator_Util.c
REID_GEN_CFLAGS += -I"$(TILER_PATH)/include" -I"$(CNN_GEN_PATH)"

GenReidNet: $(REID_GEN_SRCS)
	gcc -o $@ $(REID_GEN_CFLAGS) $(REID_GEN_SRCS) $(TILER_LIB)

$(REID_MODEL_GEN_C): GenReidNet
	./GenReidNet

reid_model: $(REID_MODEL_GEN_C)

EXTRA_MODEL_GEN := ExtraKernels
EXTRA_MODEL_GEN_C := $(addsuffix .c, $(EXTRA_MODEL_GEN))
EXTRA_MODEL_GEN_CH := $(EXTRA_MODEL_GEN_C) $(addsuffix .h, $(EXTRA_MODEL_GEN))

GenExtraTile: $(REID_PATH)/ExtraModels.c
	gcc -o $@ -I"$(TILER_INC)" -I"$(CNN_GEN_PATH)" $(REID_PATH)/ExtraModels.c $(TILER_LIB)

$(EXTRA_MODEL_GEN_C): GenExtraTile
	./GenExtraTile

extra_model: $(EXTRA_MODEL_GEN_C)

tiler_models: reid_model extra_model

APP_SRCS := \
  main.c \
  $(REID_PATH)/reid_pipeline.c \
  $(REID_PATH)/network_process_manual.c \
  $(REID_PATH)/param_layer_manual.c \
  $(REID_PATH)/dnn_utils.c \
  $(REID_PATH)/ExtraBasicKernels.c \
  $(REID_PATH)/ImgIO.c \
  $(REID_MODEL_GEN_C) \
  $(EXTRA_MODEL_GEN_C) \
  $(CNN_KER_PATH)/CNN_Conv_BasicKernels.c \
  $(CNN_KER_PATH)/CNN_BiasReLULinear_BasicKernels.c \
  $(CNN_KER_PATH)/CNN_Pooling_BasicKernels.c
APP_INC += $(PWD) $(REID_PATH) "$(TILER_INC)" "$(CNN_KER_PATH)"
APP_CFLAGS += -O3 -DNORM_ROUND -DCONFIG_GAPOC_A -DTEST_RESPONSE=test_response_l$(TEST_LEVEL)
ifeq ($(SILENT),1)
  APP_CFLAGS += -DSILENT=1
endif

build: tiler_models

clean::
	rm -f GenReidNet GenExtraTile
	rm -f $(REID_MODEL_GEN_CH) $(EXTRA_MODEL_GEN_CH)

.PHONY: reid_model extra_model tiler_models build clean

include $(RULES_DIR)/pmsis_rules.mk
